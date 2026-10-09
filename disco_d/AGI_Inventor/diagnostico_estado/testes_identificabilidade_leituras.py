"""Consistência em ciclos e cruzamentos; não medir ganho de aprendizagem aqui."""

import numpy as np

from diagnosticar_identificabilidade_leituras import crossings, word_potential


def main():
    a, b, c, d = (0,), (1,), (2,), (3,)
    assert word_potential([(a, b), (c, d)], [3., -7.])["program_only_potential_can_interpolate_this_finite_sample"]
    assert word_potential([(a, b), (b, c), (a, c)], [1., 2., 3.])["program_only_potential_can_interpolate_this_finite_sample"]
    assert not word_potential([(a, b), (b, c), (a, c)], [1., 2., 4.])["program_only_potential_can_interpolate_this_finite_sample"]
    assert not word_potential([(a, a)], [1.])["program_only_potential_can_interpolate_this_finite_sample"]
    plans = np.asarray([[[0], [1]], [[1], [0]]])
    same = crossings(plans, np.asarray([1., -1.]), 1)
    assert same["repeated_pair_groups"] == 1 and same["repeated_pairs_with_different_returns"] == 0
    opposite = crossings(plans, np.asarray([1., 1.]), 1)
    assert opposite["repeated_pairs_requiring_opposite_decisions"] == 1
    assert not opposite["finite_word_potential"]["program_only_potential_can_interpolate_this_finite_sample"]
    tied = crossings(plans, np.zeros(2), 1)
    assert tied["nonzero_real_contrasts"] == 0 and tied["repeated_pairs_requiring_opposite_decisions"] == 0
    print("identificabilidade_ok: arvore, ciclo_consistente, ciclo_contraditorio, orientacao_e_empate")


if __name__ == "__main__":
    main()
