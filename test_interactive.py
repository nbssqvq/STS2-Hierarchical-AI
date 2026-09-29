import requests
import json
import time

BASE_URL = "http://localhost:15526"
STATE_URL = "/api/v1/singleplayer"
ACTION_URL = "/api/v1/singleplayer"

def get_state():
    resp = requests.get(f"{BASE_URL}{STATE_URL}", timeout=5)
    resp.raise_for_status()
    return resp.json()

def post_action(payload):
    resp = requests.post(f"{BASE_URL}{ACTION_URL}", json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()

def print_state(state):
    print("\n" + "=" * 60)
    print("current state:")
    print(" " * 60)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print(" " * 60)

def main():
    print("STS2 interaction test")
    print(f"localhttp: {BASE_URL}")
    print()

    try:
        state = get_state()
        print_state(state)
    except Exception as e:
        print(f"failed to get state: {e}")
        return

    while True:
        try:
            cmd = input("\nyour choice: ").strip()
            if not cmd:
                continue
            if cmd.lower() == "quit":
                print("exited")
                break
            if cmd.lower() == "state":
                state = get_state()
                print_state(state)
                continue

            try:
                payload = json.loads(cmd)
            except json.JSONDecodeError as e:
                print(f"parse error: {e}")
                continue

            print(f"executed: {json.dumps(payload, ensure_ascii=False)}")
            try:
                result = post_action(payload)
                print("result:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"execution failed: {e}")
                continue

            time.sleep(0.5)
            try:
                state = get_state()
                print_state(state)
            except Exception as e:
                print(f"refresh state failed: {e}")

        except KeyboardInterrupt:
            print("\nquitting")
            break
        except Exception as e:
            print(f"unexpected error: {e}")

if __name__ == "__main__":
    main()