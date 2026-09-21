import os
from broker_config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "BOT_forex", "mt5", "config", "mt5.ini"
)

CONTENT = f"""[Common]
Login={MT5_LOGIN}
Password={MT5_PASSWORD}
Server={MT5_SERVER}
"""


def main() -> None:
    with open(OUTPUT_PATH, "w") as f:
        f.write(CONTENT)
    print(f"✅ mt5.ini updated at {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()