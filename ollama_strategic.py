from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import requests

from llm_strategic import StrategicAgent
from project_config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL
from reward_function import combat_reward_profile
from strategic_routing import StrategicPolicyGuard


ACTION_NAMES = [
    "choose_map_node", "shop_purchase", "choose_event_option", "advance_dialogue",
    "choose_rest_option", "select_card_reward", "skip_card_reward", "select_card",
    "confirm_selection", "select_relic", "select_bundle", "confirm_bundle_selection",
    "combat_select_card", "combat_confirm_selection", "proceed", "none",
]

ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTION_NAMES},
        "index": {"type": "integer"},
        "card_index": {"type": "integer"},
    },
    "required": ["action"],
    "additionalProperties": False,
}

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["act", "finished"]},
        "action": ACTION_SCHEMA,
        "route_id": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["status", "action", "reason"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are the strategic agent for Slay the Spire 2.
PPO exclusively controls ordinary combat card play. You control meaningful choices outside
ordinary combat, plus selection modals opened by combat. Return JSON matching the supplied
schema and never invent an action or index. Use status=act for the next action. Use
status=finished only when all useful decisions on the screen are complete and it is correct to leave.
When decision_context is combat_interstitial, the choice directly affects the current battle;
make the choice using the hand, enemies and prompt, then control returns to PPO.
In a shop, buy useful affordable items before returning finished. can_proceed means leaving is
possible, not that leaving is strategically correct. The action object must use its `action`
field and one exact name from allowed_actions; never use fields such as type or action_name.
For status=finished, set the unused action to {"action":"none"}. Every response
must include the top-level reason field, even for an obvious choice. Keep it under
20 Chinese characters and cite only visible state. Example:
{"status":"act","action":{"action":"choose_event_option","index":0},"reason":"生命不足，避免失血"}
Output JSON only."""


GLOBAL_STRATEGY = """策略优先级：硬规则 > 当前生存例外 > 一般优先级。
不得选择会立即死亡、卡死流程或无法支付的选项。不确定时选择保守、少损失生命和少浪费资源的方案。
只根据输入中明确存在的信息决策；缺少牌组、下一节点或Boss距离时不得自行编造。
若已知牌组超过20张，非核心、非补缺牌应跳过。Boss临近时尽量保留已有关键药水；药水不足时不要假设拥有两瓶。"""


STRATEGY_BY_SCREEN: Dict[str, str] = {
    "map": """地图：full_map.nodes 是从当前位置可达的完整路线图，children 给出连接。route_candidates 列出从本次可选房间通往终点的完整路线；首次选路时应选一条路线，action.index 使用该路线的 first_action_index，并在输出顶层填入其 route_id。之后会沿计划路线前进，只有生命骤降、路线不可走或出现紧急危险时才重新规划。
生命低于40%且可选路线含精英/Boss时，必须选篝火或更安全路线。通常避开精英。
一般优先：篝火 > 商店 > 事件 > 普通战斗 > 精英，并偏好战斗少、篝火多的路线。
生命高于70%、已有成型协同且有回复资源时才主动打精英。
若都危险，选择战斗最少、预计生命损失最低的路线。""",
    "map_screen": """地图：full_map.nodes 是从当前位置可达的完整路线图，children 给出连接。选择 route_candidates 中一条完整路线，action.index 使用其 first_action_index，并在输出顶层填入 route_id；只有紧急危险或路线失效时才改线。
生命低于40%且可选路线含精英/Boss时，必须选篝火或更安全路线。通常避开精英。
一般优先：篝火 > 商店 > 事件 > 普通战斗 > 精英。生命高于70%且有成型协同与回复资源时才主动打精英。""",
    "card_reward": """卡牌奖励：废牌不如跳过；稀有度不代表价值。优先核心协同牌，其次补足输出、防御、过牌或能量缺口。
牌组超过20张时，只拿明确核心牌。不要重复拿弱过渡攻击/防御。输出严重不足可拿一张优质攻击；生命很低且防御不足可拿关键防御。
铁甲战士优先维持单一主轴：力量（恶魔形态/重刃/旋风斩/祭品）、格挡（全身撞击/耸肩无视/无惧痛苦）或消耗（无惧痛苦配消耗牌）。不要无过牌堆积攻击或消耗牌。""",
    "shop": """商店是连续决策。shop_visit.purchases 是本次访问已成功购买的物品；每次只返回下一项动作，购买后根据刷新后的金币和库存重新决策，完成所有有价值购买后才返回 finished。
硬规则：只要 card_removal 有库存且能支付，必须在购买其他物品前先买 card_removal；删牌优先不会自动消失的诅咒，其次根据数量调整打击和防御，尽量维持防御比打击多2张。同一种基础牌有多张时必须优先删除未升级版本；只要仍有未升级的打击，就不得删除打击+，防御同理。不得购买售罄、买不起的物品；药水槽满时不得购买药水。
删牌完成后，优先核心牌、高价值能量/过牌/防御遗物、保命或爆发药水。一般建议保留150金币给未来商店，但这是软性建议，不得因此放弃当前可支付的删牌，也不应覆盖明显高价值或生存必需的购买。
不要仅因为 can_proceed=true 就离开。""",
    "shop_screen": """商店：可支付时先删牌，随后考虑核心牌、能量/过牌/防御遗物和保命药水。建议保留150金币，但这不是强制限制。不得购买售罄或买不起的物品。""",
    "fake_merchant": """商店：只买对当前体系明确有帮助的高价值物品，避免为普通攻击增益花光金币。""",
    "rest_site": """篝火决策必须先计算 hp_ratio = hp / max_hp，并比较休息的实际收益与其他选项的长期收益。
hp_ratio < 60%：优先休息。
hp_ratio 在60%-75%：若下一场是精英/Boss或缺少回复手段，优先休息；否则考虑升级。
hp_ratio > 75%：强烈倾向升级或选择其他永久成长选项。此时休息会浪费部分恢复量，通常价值低。仅当没有任何有价值的非休息选项，或下一场明确存在极高死亡风险时，才允许休息。
选择升级时：核心能力牌/核心攻击牌 > 关键防御或过牌 > 普通牌。铁甲优先当前主轴中的恶魔形态、全身撞击、无惧痛苦等核心牌。
不得仅因为“休息”是第一个选项或最安全选项就选择休息。必须根据 options 中的名称和效果判断，不能假设 index=0 总是正确。""",
    "rest": """篝火：先计算 hp_ratio。低于60%优先休息；60%-75%仅在下一场是精英/Boss或缺少回复时优先休息，否则考虑升级；高于75%强烈倾向升级或其他永久成长，只有没有有价值的非休息选项或下一场死亡风险极高时才休息。不得默认选择 index=0。""",
    "campfire": """篝火：先计算 hp_ratio。低于60%优先休息；60%-75%仅在下一场是精英/Boss或缺少回复时优先休息，否则考虑升级；高于75%强烈倾向升级或其他永久成长，只有没有有价值的非休息选项或下一场死亡风险极高时才休息。不得默认选择 index=0。""",
    "fire": """篝火：先计算 hp_ratio。低于60%优先休息；60%-75%仅在下一场是精英/Boss或缺少回复时优先休息，否则考虑升级；高于75%强烈倾向升级或其他永久成长，只有没有有价值的非休息选项或下一场死亡风险极高时才休息。不得默认选择 index=0。""",
    "event": """事件：不得选择会立即死亡的代价。先比较支付生命后的当前生命与接下来几场战斗的风险，再比较长期收益。
若当前生命健康、支付后仍有足够生存余量，优先考虑“失去生命，但每完成一场战斗增加最大生命值”这类持续成长选项；它会随战斗次数累积最大生命，也可能提高以后按最大生命比例计算的回复量。不要仅因另一选项立即回血和升级一张牌就忽略这一长期收益。每次重复扣血都必须重新计算下一次代价；若支付后生命低于最大生命的50%或低于30点，强烈倾向停止继续扣血，选择离开或安全选项。未来回复潜力不能代替当前生命。若下一战危险或本幕剩余战斗很少，也优先即时回血和升级。
生命高于60%才考虑用生命换金币/强遗物；生命高于50%才考虑用生命换最大生命，但仍要检查支付后的生命。金币超过150且遗物很强时可换。
通常拒绝重大且难移除的诅咒；若诅咒会自动消失，或可立即删除且收益为体系核心遗物，可接受。""",
    "event_screen": """事件：不得选择会立即死亡的代价。生命健康且支付后仍安全时，优先考虑每场战斗增加最大生命值的持续成长；最大生命提高也可能增加以后按比例计算的回复量。重复扣血时每次重算下一次代价；若支付后生命低于最大生命的50%或低于30点，强烈倾向停止并选择离开或安全选项。生命偏低、下一战危险或剩余战斗少时，优先即时回血。生命高于60%才考虑用生命换一般资源。通常拒绝难移除的重大诅咒，除非收益能改变体系且有可靠移除手段。""",
    "hand_select": """这是战斗中选择，结果直接影响当前战斗。先读 prompt 判断本次操作是消耗、升级、发现/获得，还是保留/弃置；这些操作的选牌优先级不能混用。
消耗/移出本场战斗：若候选牌描述明确写明被消耗后触发有利效果，优先消耗该牌；否则选择本场战斗中价值最低、当前不需要的牌，通常先消耗弱基础牌，保留核心牌、过牌、关键攻击和保命牌。消耗不是永久删牌，不要套用商店删牌的卡组数量规则。若某张格挡牌能避免本回合死亡，或某张攻击牌能立即击杀危险敌人，应保留它；结合敌人意图和当前生命判断，而非只看牌名或候选顺序。
升级：选择升级后收益最大的当前主轴核心牌，优先费用下降、过牌/能量、关键能力或攻击效果；普通打击和防御通常最后考虑。发现/获得：选择当前战斗最有用的核心协同、保命、过牌或能量牌。保留/弃置：保留下一回合需要的核心能力、爆发攻击和保命牌，优先弃置暂时无用的基础牌。
丢弃：若候选牌描述明确写明被丢弃后触发有利效果，优先丢弃该牌；否则优先丢弃当前战斗价值低的牌。只按此次操作匹配触发条件，不要把“被丢弃时”的效果当成“被消耗时”也会触发。
如果 prompt 的含义与上述类别不一致，以 prompt 的实际操作为准；不要把“选择要消耗或丢弃的牌”当作“选择要保留或升级的牌”。""",
    "card_select": """牌组选择：先根据 prompt 和 screen_meta.screen_type 判断是在删牌、升级还是获得卡牌。删牌时优先不会自动消失的诅咒；否则统计基础打击和基础防御的数量，尽量维持防御比打击多2张，再决定删除哪一类。同一种基础牌有多张时必须优先删除未升级版本：只要仍有未升级的打击就不得删除打击+，只要仍有未升级的防御就不得删除防御+。
升级时逐张比较当前 cost、star_cost、description 与 upgrade_preview 的变化，并结合牌组主轴判断实际收益。优先明显降低费用、增强抽牌/能量、强化核心机制或解决当前短板的升级；普通打击和防御通常最后考虑，除非没有更有价值的升级或有明确的当前需求。upgrade_preview 缺失时不得编造升级效果，也不要把候选 index=0 当成优先级。获得卡牌时优先当前流派核心牌。""",
    "card_grid_selection": """牌组选择：先根据 prompt 和 screen_meta.screen_type 判断是在删牌、升级还是获得卡牌。删牌时优先不会自动消失的诅咒；否则统计基础打击和基础防御的数量，尽量维持防御比打击多2张，再决定删除哪一类。同一种基础牌有多张时必须优先删除未升级版本：只要仍有未升级的打击就不得删除打击+，只要仍有未升级的防御就不得删除防御+。
升级时逐张比较当前 cost、star_cost、description 与 upgrade_preview 的变化，并结合牌组主轴判断实际收益。优先明显降低费用、增强抽牌/能量、强化核心机制或解决当前短板的升级；普通打击和防御通常最后考虑，除非没有更有价值的升级或有明确的当前需求。upgrade_preview 缺失时不得编造升级效果，也不要把候选 index=0 当成优先级。获得卡牌时优先当前流派核心牌。""",
    "choose_a_card": """选牌：优先当前流派核心，其次补足当前最严重的输出、防御、过牌或能量缺口；无合适牌时跳过。""",
    "relic_select": """遗物通常优先：能量 > 过牌 > 防御 > 攻击，但必须考虑当前牌组协同。能改变体系的核心遗物优先。""",
    "bundle_select": """组合奖励：选择与当前主轴协同且不会显著增加废牌的组合，优先稳定生存和过牌。""",
}


def strategy_prompt(observation: Dict[str, Any]) -> str:
    screen = str(observation.get("screen", ""))
    section = STRATEGY_BY_SCREEN.get(screen, "")
    prompt = f"{GLOBAL_STRATEGY}\n当前界面策略：\n{section}" if section else GLOBAL_STRATEGY
    if screen not in {"event", "event_screen"}:
        return prompt
    prompt += (
        "\n本类事件的停止线优先于前文一般长期收益建议：对‘扣当前生命换每场战斗增加最大生命’，"
        "仅当当前HP高于最大生命60%，且支付后HP仍至少为max(最大生命60%, 30点)时才可选择。"
        "否则停止继续支付，优先即时回复、升级或其他无生命代价选项；最大生命成长可以放弃，不要反复投入到低血量。"
    )
    player = observation.get("player", {})
    candidates = observation.get("candidates", {})
    if not isinstance(player, dict) or not isinstance(candidates, dict):
        return prompt
    hp = float(player.get("hp") or 0)
    max_hp = float(player.get("max_hp") or 0)
    if max_hp <= 0:
        return prompt
    notes = []
    for option in candidates.get("options", []):
        if not isinstance(option, dict):
            continue
        cost = StrategicPolicyGuard._event_hp_cost(option)
        if cost <= 0:
            continue
        remaining = hp - cost
        max_hp_growth = StrategicPolicyGuard._event_max_hp_growth(option)
        if remaining <= 0:
            risk = "会死亡，绝不可选"
        elif max_hp_growth and (hp / max_hp < 0.60 or remaining < max(30.0, 0.60 * max_hp)):
            risk = "达到最大生命事件停止线，改选回复或升级"
        elif remaining < 30 or remaining / max_hp < 0.50:
            risk = "生命过低，强烈建议停止扣血"
        else:
            risk = "仍需考虑下一战风险"
        note = f"选项{option.get('index')}：支付{cost}生命后剩余{remaining:g}/{max_hp:g}，{risk}"
        if max_hp_growth:
            note += "；最大生命成长可放弃，不要为反复叠加降到停止线以下"
        notes.append(note)
    if notes:
        prompt += "\n本次事件代价核算（只评估当前这一次，下一次须重新计算）：\n" + "；".join(notes)
    return prompt


class OllamaStrategicAgent:
    """Structured Ollama strategy with a deterministic, non-blocking fallback."""

    requires_freshness_check = True

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL,
                 timeout: Optional[float] = None, temperature: float = 0.1,
                 fallback: Optional[StrategicAgent] = None, required: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        configured_timeout = timeout if timeout is not None else os.environ.get(
            "STS2_STRATEGIC_TIMEOUT", "420"
        )
        self.timeout = float(configured_timeout)
        self.temperature = float(temperature)
        self.fallback = fallback or StrategicAgent()
        self.required = bool(required)
        self.session = requests.Session()
        self._availability: Optional[bool] = None
        self._availability_checked_at = 0.0
        self._reported_unavailable = False

    def _model_available(self) -> bool:
        now = time.monotonic()
        if self._availability is not None and now - self._availability_checked_at < 30.0:
            return self._availability
        self._availability_checked_at = now
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=min(3.0, self.timeout))
            response.raise_for_status()
            names = {str(item.get("name", "")) for item in response.json().get("models", [])}
            requested = self.model.removesuffix(":latest")
            self._availability = any(name == self.model or name.removesuffix(":latest") == requested for name in names)
        except (requests.RequestException, ValueError, TypeError):
            self._availability = False
        return self._availability

    def model_available(self) -> bool:
        """Public preflight used by batch collection to reject silent fallback runs."""
        return self._model_available()

    def choose_combat_rewards(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Choose bounded A/B/C once before a battle; D remains type-specific."""
        combat_type = str(state.get("state_type") or state.get("screen") or "").lower()
        defaults = combat_reward_profile(combat_type)
        hp_guidance = (
            "这是Boss战：以击败Boss为首要目标，可把HP当作换取胜利的资源，不要为了保留战后HP而降低战斗表现；"
            "C必须在0到0.1之间，优先设为0。固定死亡惩罚仍然有效。"
            if combat_type == "boss" else
            "普通怪和精英战的C必须在0.8到1.2之间，按战斗风险小幅调整。"
        )
        if not self._model_available():
            if self.required:
                raise RuntimeError(f"Required strategic model unavailable: {self.model}")
            print(f"[COMBAT_REWARD] model unavailable; fixed coefficients for {combat_type}", flush=True)
            return defaults
        battle = state.get("battle", {})
        player = state.get("player", {})
        context = {
            "type": combat_type,
            "floor": state.get("run", {}).get("floor", 0),
            "hp": player.get("hp"), "max_hp": player.get("max_hp"),
            "potions": player.get("potions", []),
            "enemies": [{"name": enemy.get("name"), "hp": enemy.get("hp"),
                         "intent": enemy.get("intent")}
                        for enemy in battle.get("enemies", []) if isinstance(enemy, dict)],
            "strategic_context": state.get("strategic_context", {}),
            "defaults": defaults,
        }
        route_guidance = (
            "若路线信息显示战斗后1步就是篝火，且当前HP并非濒危，可考虑把C轻微调低，"
            "因为之后有恢复机会；不得因此接受高死亡风险。路线更远或未知时按通常风险评估。"
            if context["strategic_context"].get("next_rest_steps") == 1 else
            "路线中的篝火若较远，不应显著降低本场HP成本；路线未知时按通常风险评估。"
        )
        payload = {
            "model": self.model, "stream": False, "think": False, "keep_alive": "30m",
            "format": {
                "type": "object",
                "properties": {
                    "A": {"type": "number"},
                    "B": {"type": "number"},
                    "C": {"type": "number"},
                },
                "required": ["A", "B", "C"],
                "additionalProperties": False,
            },
            "options": {"temperature": 0.1, "num_ctx": 2048, "num_predict": 96},
            "messages": [
                {"role": "system", "content": (
                    "你只负责为即将开始的这场杀戮尖塔战斗选择奖励系数。"
                    "胜利奖励 A+B*层数，战斗期间每损失1点净HP扣C；药水扣分D由程序固定。"
                    "根据敌人难度调整A/B/C，不要改D。" + hp_guidance + route_guidance +
                    "B必须在0.4到0.6；"
                    "A范围：普通怪5到20，精英18到40，Boss40到75。"
                    "只返回JSON对象，例如 {\"A\":50,\"B\":0.5,\"C\":0.0}。")},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
            ],
        }
        started = time.monotonic()
        try:
            response = self.session.post(f"{self.base_url}/api/chat", json=payload,
                                         timeout=self.timeout)
            response.raise_for_status()
            choice = json.loads(response.json().get("message", {}).get("content", ""))
            profile = combat_reward_profile(combat_type, choice)
            print(f"[COMBAT_REWARD_LLM] type={combat_type} latency={time.monotonic()-started:.2f}s "
                  f"raw={choice} chosen={profile}", flush=True)
            return profile
        except (requests.RequestException, ValueError, TypeError) as exc:
            if self.required:
                raise RuntimeError(f"Required strategic model request failed: {exc}") from exc
            print(f"[COMBAT_REWARD] LLM failed: {exc}; fixed coefficients for {combat_type}", flush=True)
            return defaults

    def plan_route(self, observation: Dict[str, Any], raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """Choose one complete route at the first meaningful map fork."""
        routes = observation.get("route_candidates", [])
        if not routes:
            return self.decide_observation(observation, raw_state)
        if not self._model_available():
            return self.decide_observation(observation, raw_state)
        player = observation.get("player", {})
        request_payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "format": {
                "type": "object",
                "properties": {
                    "route_id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["route_id", "reason"],
                "additionalProperties": False,
            },
            "options": {"temperature": self.temperature, "num_ctx": 4096, "num_predict": 128},
            "messages": [
                {"role": "system", "content": (
                    "Choose one complete connected route for the current act. "
                    "Prefer survival, rest sites and useful shops; avoid elites when weak. "
                    "Use current HP, gold, deck and the entire listed route, not just its first room. "
                    "Return only a JSON object of the form "
                    "{\"route_id\": integer, \"reason\": \"brief Chinese reason\"}. "
                    "The reason must cite visible HP and route facts, at most 20 Chinese characters."
                )},
                {"role": "user", "content": json.dumps({
                    "run": observation.get("run"),
                    "player": {key: player.get(key) for key in (
                        "character", "hp", "max_hp", "gold", "deck", "relics", "potions",
                    )},
                    "routes": routes,
                }, ensure_ascii=False, default=str)},
            ],
        }
        started = time.monotonic()
        try:
            response = self.session.post(f"{self.base_url}/api/chat", json=request_payload,
                                         timeout=self.timeout)
            response.raise_for_status()
            choice = json.loads(response.json().get("message", {}).get("content", ""))
            route_id = choice.get("route_id") if isinstance(choice, dict) else None
            route = next((item for item in routes if item["id"] == route_id), None)
            if route is not None:
                reason = str(choice.get("reason") or "").strip()[:80]
                print(f"[LLM_ROUTE] id={route_id} rooms={len(route['rooms'])} "
                      f"latency={time.monotonic() - started:.2f}s reason={reason}", flush=True)
                return {"status": "act", "action": {
                    "action": "choose_map_node", "index": route["first_action_index"],
                }, "route_id": route_id, "reason": reason, "source": "ollama"}
            print(f"[LLM_ROUTE] invalid route_id={route_id}; using normal decision", flush=True)
        except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
            if self.required:
                raise RuntimeError(f"Required strategic route request failed: {exc}") from exc
            print(f"[LLM_ROUTE] request failed: {exc}; using normal decision", flush=True)
        return self.decide_observation(observation, raw_state)

    @staticmethod
    def _parse_message(payload: Dict[str, Any]) -> Dict[str, Any]:
        content = payload.get("message", {}).get("content", "")
        if isinstance(content, dict):
            decision = content
        else:
            decision = json.loads(str(content))
        if not isinstance(decision, dict):
            raise ValueError("Ollama decision is not an object")
        status = decision.get("status")
        if status not in {"act", "finished"}:
            raise ValueError(f"Unsupported decision status: {status}")
        if status == "act" and not isinstance(decision.get("action"), dict):
            raise ValueError("status=act requires an action object")
        decision["reason"] = str(decision.get("reason") or "").strip()[:80]
        return decision

    def decide_observation(self, observation: Dict[str, Any], raw_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self._model_available():
            if self.required:
                raise RuntimeError(f"Required strategic model unavailable: {self.model}")
            if not self._reported_unavailable:
                print(f"[LLM_FALLBACK] Ollama model unavailable: {self.model}", flush=True)
                self._reported_unavailable = True
            return {"status": "act", "action": self.fallback.decide(raw_state),
                    "reason": "ollama_unavailable", "source": "fallback"}

        request_payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "format": DECISION_SCHEMA,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 4096,
                "num_predict": 128,
            },
            "messages": [
                {"role": "system", "content": f"{SYSTEM_PROMPT}\n{strategy_prompt(observation)}"},
                {"role": "user", "content": json.dumps(observation, ensure_ascii=False, default=str)},
            ],
        }
        started = time.monotonic()
        error: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = self.session.post(f"{self.base_url}/api/chat", json=request_payload, timeout=self.timeout)
                response.raise_for_status()
                decision = self._parse_message(response.json())
                decision["source"] = "ollama"
                print(f"[LLM_DECISION] context={observation.get('decision_context')} "
                      f"screen={observation.get('screen')} status={decision['status']} "
                      f"latency={time.monotonic() - started:.2f}s "
                      f"reason={decision['reason']}", flush=True)
                return decision
            except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
                error = exc
                response_body = ""
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    response_body = f" body={exc.response.text[:1000]}"
                print(f"[LLM_RETRY] attempt={attempt + 1} error={exc}{response_body}", flush=True)
                # A timed-out generation may still be running on the server. Retrying
                # immediately only queues another expensive request behind it.
                if (isinstance(exc, requests.ReadTimeout)
                        or (isinstance(exc, requests.HTTPError)
                            and exc.response is not None and exc.response.status_code == 503)):
                    break
        # Strict runs stop here so the watchdog can save and recover without
        # adding fallback decisions to the PPO data.
        if self.required:
            raise RuntimeError(f"Required strategic model request failed: {error}") from error
        print(f"[LLM_FALLBACK] request failed after attempts: {error}", flush=True)
        return {"status": "act", "action": self.fallback.decide(raw_state),
                "reason": "ollama_request_failed", "source": "fallback"}

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility entry point for callers without a strategic observation."""
        return self.fallback.decide(state)

    def fallback_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.fallback.decide(state)

    def note_action_result(self, state: Dict[str, Any], payload: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Keep deterministic fallback state aligned with successfully applied actions."""
        self.fallback.note_action_result(state, payload, response)

    def close(self) -> None:
        self.session.close()


def create_strategic_agent() -> OllamaStrategicAgent:
    """Require the configured model and optionally forbid runtime fallback."""
    required = os.environ.get("STS2_STRATEGIC_REQUIRED", "").strip().lower() in {"1", "true", "yes"}
    allow_runtime_fallback = os.environ.get(
        "STS2_STRATEGIC_ALLOW_FALLBACK", "1"
    ).strip().lower() in {"1", "true", "yes"}
    agent = OllamaStrategicAgent(required=required)
    if required:
        if not agent.model_available():
            agent.close()
            raise RuntimeError(f"Required strategic model unavailable at startup: {agent.model}")
        agent.required = not allow_runtime_fallback
        runtime_mode = "enabled" if allow_runtime_fallback else "disabled"
        print(f"[LLM_PREFLIGHT] model={agent.model} available; "
              f"runtime fallback {runtime_mode}", flush=True)
    return agent


__all__ = ["DECISION_SCHEMA", "OllamaStrategicAgent", "create_strategic_agent"]
