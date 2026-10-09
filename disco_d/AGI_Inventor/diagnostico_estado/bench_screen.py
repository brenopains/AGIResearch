"""Mede custo de leitura de quadro para planejar coleta limitada."""

from time import perf_counter

from coleta import gray, make_ale


if __name__ == "__main__":
    ale = make_ale("breakout", 17)
    for name, method in (("weighted_rgb", lambda: gray(ale)),
                         ("ale_grayscale", lambda: ale.getScreenGrayscale())):
        start = perf_counter()
        for _ in range(1000):
            method()
        print(name, (perf_counter() - start) / 1000)
    action = list(ale.getMinimalActionSet())[0]
    start = perf_counter()
    for _ in range(1000):
        ale.act(action)
        if ale.game_over():
            ale.reset_game()
    print("ale_act", (perf_counter() - start) / 1000)
    state = ale.cloneSystemState()
    start = perf_counter()
    for _ in range(1000):
        ale.restoreSystemState(state)
        state = ale.cloneSystemState()
    print("clone_restore", (perf_counter() - start) / 1000)
