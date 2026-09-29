using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.IO;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace PunchOffInstantFix;

[ModInitializer(nameof(Initialize))]
public static class Patch
{
    private const string Id = "local.sts2.punchoff.instantwait.v1";
    private static readonly MethodInfo OriginalWait = AccessTools.Method(typeof(Cmd), nameof(Cmd.Wait), [typeof(float), typeof(bool)]);
    private static readonly MethodInfo ReplacementWait = AccessTools.Method(typeof(Patch), nameof(Wait));
    private static bool _installed;
    private static bool _instantModeApplied;
    private static long _waitCalls;

    public static void Initialize()
    {
        if (_installed) return;
        // Mod initializers can run before SaveManager has finished loading.
        // Defer preference access until the SceneTree has advanced.
        _ = RetryInstantModeAfterStartup();
        var method = AccessTools.Method(typeof(PunchOff), "PunchEachOther");
        var stateMachine = method?.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType;
        var moveNext = stateMachine == null ? null : AccessTools.Method(stateMachine, "MoveNext");
        if (moveNext == null) throw new InvalidOperationException("PunchOff async state machine not found; unsupported game build.");
        new Harmony(Id).Patch(moveNext, transpiler: new HarmonyMethod(typeof(Patch), nameof(Transpile)));
        _installed = true;
        GD.Print("[PunchOffInstantFix] Installed: four local Cmd.Wait calls replaced; Instant Mode is enforced.");
#if TEST_DRIVER
        _ = Probe.Run();
#endif
    }

    public static IEnumerable<CodeInstruction> Transpile(IEnumerable<CodeInstruction> instructions)
    {
        var code = instructions.ToList();
        var calls = code.Where(i => i.Calls(OriginalWait)).ToList();
        // Fail rather than apply a partial patch to an unrecognized game build.
        if (calls.Count != 4) throw new InvalidOperationException($"Expected 4 PunchOff waits, found {calls.Count}.");
        foreach (var call in calls) { call.opcode = OpCodes.Call; call.operand = ReplacementWait; }
        return code;
    }

    public static Task Wait(float seconds, bool ignoreCombatEnd)
    {
        EnsureInstantMode();
        if (NonInteractiveMode.IsActive || SaveManager.Instance.PrefsSave.FastMode != FastModeType.Instant)
            return Cmd.Wait(seconds, ignoreCombatEnd);
        return RealTimeWait(seconds);
    }

    private static void EnsureInstantMode()
    {
        if (_instantModeApplied) return;
        try
        {
            var saveManager = SaveManager.Instance;
            var prefs = saveManager.PrefsSave;
            if (prefs.FastMode != FastModeType.Instant)
            {
                prefs.FastMode = FastModeType.Instant;
                saveManager.SavePrefsFile();
                GD.Print("[STS2_RL_PREFLIGHT] Instant Mode enabled and saved.");
            }
            else
            {
                GD.Print("[STS2_RL_PREFLIGHT] Instant Mode already enabled.");
            }

            var markerPath = Path.Combine(
                System.Environment.GetFolderPath(System.Environment.SpecialFolder.ApplicationData),
                "SlayTheSpire2", "sts2_rl_instant_mode.json");
            Directory.CreateDirectory(Path.GetDirectoryName(markerPath)!);
            File.WriteAllText(markerPath,
                $"{{\"pid\":{System.Environment.ProcessId},\"mode\":\"Instant\",\"utc\":\"{DateTimeOffset.UtcNow:O}\"}}");
            _instantModeApplied = true;
        }
        catch (Exception ex)
        {
            // Initialization can precede save loading on some game builds. The
            // first intercepted game wait retries this after startup completes.
            if (!_instantModeApplied)
                GD.PrintErr($"[STS2_RL_PREFLIGHT] Could not enable Instant Mode yet: {ex.Message}");
        }
    }

    private static async Task RetryInstantModeAfterStartup()
    {
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree) return;
            for (var attempt = 0; attempt < 60 && !_instantModeApplied; attempt++)
            {
                var timer = tree.CreateTimer(attempt == 0 ? 3.0 : 1.0, processAlways: true,
                    processInPhysics: false, ignoreTimeScale: true);
                await tree.ToSignal(timer, SceneTreeTimer.SignalName.Timeout);
                EnsureInstantMode();
            }
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2_RL_PREFLIGHT] Instant Mode startup retry stopped: {ex.Message}");
        }
    }

    private static async Task RealTimeWait(float seconds)
    {
        long count = Interlocked.Increment(ref _waitCalls);
        // A SceneTree timer signals on the game thread; no blocking sleep or
        // background-thread node operations. Preserve the original 0.1/1.2s durations.
        var tree = (SceneTree)Engine.GetMainLoop();
        var timer = tree.CreateTimer(Math.Max(0.01, seconds), processAlways: true,
                                    processInPhysics: false, ignoreTimeScale: true);
        if (count <= 4) GD.Print($"[PunchOffInstantFix] Real-time wait #{count}: {seconds}s, Instant Mode active.");
        await tree.ToSignal(timer, SceneTreeTimer.SignalName.Timeout);
    }
}
