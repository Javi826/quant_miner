import logging

logger = logging.getLogger("pipeline.step0")


def run(config: dict) -> bool:
    selected_symbols = config.get("selected_symbols", [])
    if not selected_symbols:
        logger.warning("⚠ No symbols selected. Aborting.")
        return False
    logger.info(f"📋 Symbol mode: MANUAL — {len(selected_symbols)} symbol(s): {selected_symbols}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _config = {"selected_symbols": ["EURUSD", "GBPUSD"]}
    run(_config)