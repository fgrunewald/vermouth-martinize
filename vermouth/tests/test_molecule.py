# Copyright 2018 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import pytest

import copy
import itertools
import numpy as np
import pytest
import vermouth
import vermouth.molecule
from vermouth.molecule import Interaction, Molecule


@pytest.fixture
def molecule():
    molecule = vermouth.molecule.Molecule()
    molecule.meta['test'] = True
    molecule.meta['test_mutable'] = [0, 1, 2]
    # The node keys should not be in a sorted order as it would mask any issue
    # due to the keys being accidentally sorted.
    molecule.add_node(2, atomname='CC')
    molecule.add_node(0, atomname='AA', mutable=[7, 8, 9])
    molecule.add_node(1, atomname='BB')
    molecule.add_edge(0, 1)
    molecule.add_edge(0, 2)
    molecule.add_interaction(
        type_='bonds',
        atoms=(0, 1),
        parameters=['1', '2'],
        meta={'unmutable': 0, 'mutable': [4, 5, 6]},
    )
    molecule.add_interaction(
        type_='bonds',
        atoms=(0, 2),
        parameters=['a', 'b'],
    )
    return molecule


@pytest.fixture
def molecule_copy(molecule):
    return molecule.copy(as_view=False)


@pytest.fixture
def molecule_subgraph(molecule):
    return molecule.subgraph([2, 0])


@pytest.mark.xfail(reason='issue #61')
def test_copy(molecule, molecule_copy):
    assert molecule_copy is not molecule
    assert molecule_copy.meta == molecule.meta
    assert list(molecule_copy.nodes) == list(molecule.nodes)
    assert list(molecule_copy.nodes.values()) == list(molecule.nodes.values())
    assert molecule_copy.interactions == molecule.interactions


def test_copy_meta_mod(molecule, molecule_copy):
    molecule_copy.meta['test'] = False
    assert molecule_copy.meta['test'] != molecule.meta['test']
    # We are doing a copy, not a deep copy.
    assert molecule_copy.meta['test_mutable'] is molecule.meta['test_mutable']


def test_copy_node_mod(molecule, molecule_copy):
    molecule_copy.nodes[0]['atomname'] = 'mod'
    assert molecule_copy.nodes[0]['atomname'] != molecule.nodes[0]['atomname']
    extra_value = 'a new attribute'
    molecule_copy.nodes[0]['extra'] = extra_value
    assert molecule_copy.nodes[0]['extra'] == extra_value
    assert 'extra' not in molecule.nodes

    # We are looking at a copy, not a deep copy
    assert molecule_copy.nodes[0]['mutable'] is molecule.nodes[0]['mutable']

    molecule_copy.add_node('new')
    assert 'new' in molecule_copy.nodes
    assert 'new' not in molecule


def test_copy_edge_mod(molecule, molecule_copy):
    molecule_copy.add_edge(1, 2)
    assert (1, 2) in molecule_copy.edges
    assert (1, 2) not in molecule.edges
    molecule_copy.edges[(0, 1)]['attribute'] = 1
    assert molecule_copy.edges[(0, 1)]['attribute'] == 1
    assert 'attribute' not in molecule.edges[(0, 1)]


@pytest.mark.xfail(reason='issue #61')
def test_copy_interactions_mod(molecule, molecule_copy):
    molecule_copy.add_interaction(
        type_='bonds',
        atoms=(0, 2),
        parameters=['3', '4'],
        meta={'unmutable': 0},
    )
    n_bonds = len(molecule.interactions['bonds'])
    n_bonds_copy = len(molecule_copy.interactions['bonds'])
    assert n_bonds_copy > n_bonds

    molecule_copy.add_interaction(
        type_='angles',
        atoms=(0, 2, 3),
        parameters=['5', '6'],
        meta={'unmutable': 2},
    )
    assert 'angles' not in molecule.interactions


@pytest.mark.xfail(reason='issue #60')
def test_subgraph_base(molecule_subgraph):
    assert tuple(molecule_subgraph) == (2, 0)  # order matters!
    assert (0, 2) in molecule_subgraph.edges
    assert (0, 1) not in molecule_subgraph.edges  # node 1 is not there


@pytest.mark.xfail(reason='issue #61')
def test_subgraph_interactions(molecule_subgraph):
    bond_atoms = [bond.atoms for bond in molecule_subgraph.interactions['bonds']]
    assert (0, 2) in bond_atoms
    assert (0, 1) not in bond_atoms


def test_link_predicate_match():
    lp = vermouth.molecule.LinkPredicate(None)
    with pytest.raises(NotImplementedError):
        lp.match(1, 2)


@pytest.mark.parametrize('left, right, expected', (
    (  # Same
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        True,
    ),
    (  # Difference in atoms
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'notC'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Difference in parameters
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['different'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Difference in meta
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=['a', '0.2', '200'],
                            meta={'a': 0, 'other': True}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Equal with LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        True,
    ),
    (  # Different arguments for LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Different format_spec in LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(
                                    ['A', 'C'], format_spec='.2f',
                                ),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
    (  # Different LinkParameterEffector
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamDistance(['A', 'B']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        {
            'bonds': [
                Interaction(atoms=('A', 'B'),
                            parameters=[
                                'a',
                                vermouth.molecule.ParamAngle( ['A', 'B', 'C']),
                                '200',
                            ],
                            meta={'a': 0}),
                Interaction(atoms=('B', 'C'),
                            parameters=['a', '0.1', '300'],
                            meta={'b': 1}),
            ],
            'angles': [
                Interaction(atoms=('A', 'B', 'C'),
                            parameters=['1', '0.2', '200'],
                            meta={'a': 0}),
            ],
        },
        False,
    ),
))
def test_same_interactions(left, right, expected):
    """
    Test that Molecule.same_interactions works as expected.
    """
    left_mol = Molecule()
    left_mol.interactions = left
    right_mol = Molecule()
    right_mol.interactions = right
    assert left_mol.same_interactions(right_mol) == expected
    assert right_mol.same_interactions(left_mol) == expected


@pytest.mark.parametrize('left, right, expected', (
    (  # Simple identical
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        True,  # expected
    ),
    (  # Wrong order
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (1, {'c': (0, 1, 2), 'd': None}),
            (0, {'a': 'abc', 'b': 123}),
        ),
        False,  # expected
    ),

    (  # Different string
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'different', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,
    ),
    (  # Different number
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'b': 900}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Different tuple
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (3, 2, 1), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Equal Numpy array
        (  # left
            (0, {'a': np.linspace(2, 5, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': np.linspace(2, 5, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        True,  # expected
    ),
    (  # Different Numpy array
        (  # left
            (0, {'a': np.linspace(2, 5, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': np.linspace(2, 8, num=7), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Different shaped Numpy array
        (  # left
            (0, {'a': np.linspace(2, 5, num=9), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': np.linspace(2, 5, num=9).reshape((3, 3)), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Mismatch types
        (  # left
            (0, {'a': np.linspace(2, 5, num=9), 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'not an array', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),
    (  # Mismatch attribute key
        (  # left
            (0, {'a': 'abc', 'b': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        (  # right
            (0, {'a': 'abc', 'different': 123}),
            (1, {'c': (0, 1, 2), 'd': None}),
        ),
        False,  # expected
    ),

))
def test_same_nodes(left, right, expected):
    left_mol = Molecule()
    left_mol.add_nodes_from(left)
    right_mol = Molecule()
    right_mol.add_nodes_from(right)
    assert left_mol.same_nodes(right_mol) == expected
    assert right_mol.same_nodes(left_mol) == expected


@pytest.mark.parametrize('effector_class', (
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
))
@pytest.mark.parametrize('format_spec', (
    None, '.2f', '0.3f',
))
def test_link_parameter_effector_equal(effector_class, format_spec):
    """
    Test that equal LinkParameterEffector compare equal.
    """
    n_keys = effector_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(n_keys)]
    right_keys = copy.copy(left_keys)  # Let's be sure the id is different
    left = effector_class(left_keys, format_spec=format_spec)
    right = effector_class(right_keys, format_spec=format_spec)
    assert left == right


@pytest.mark.parametrize('effector_class', (
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
))
@pytest.mark.parametrize('format_right, format_left', itertools.combinations(
    (None, '.2f', '0.3f', ), 2
))
def test_link_parameter_effector_diff_format(effector_class, format_left, format_right):
    """
    Test that LinkParameterEffector compare different if they have different format.
    """
    n_keys = effector_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(n_keys)]
    right_keys = copy.copy(left_keys)  # Let's be sure the id is different
    left = effector_class(left_keys, format_spec=format_left)
    right = effector_class(right_keys, format_spec=format_right)
    assert left != right


@pytest.mark.parametrize('effector_class', (
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
))
def test_link_parameter_effector_diff_keys(effector_class):
    """
    Test that LinkParameterEffector compare different if they have different keys.
    """
    n_keys = effector_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(n_keys)]
    right_keys = ['B{}'.format(idx) for idx in range(n_keys)]
    left = effector_class(left_keys)
    right = effector_class(right_keys)
    assert left != right


@pytest.mark.parametrize('left_class, right_class', itertools.combinations((
    vermouth.molecule.ParamDistance,
    vermouth.molecule.ParamAngle,
    vermouth.molecule.ParamDihedral,
    vermouth.molecule.ParamDihedralPhase,
), 2))
def test_link_parameter_effector_diff_class(left_class, right_class):
    """
    Test that LinkParameterEffector compare different if they have different classes.
    """
    left_n_keys = left_class.n_keys_asked
    left_keys = ['A{}'.format(idx) for idx in range(left_n_keys)]
    left = left_class(left_keys)

    right_n_keys = right_class.n_keys_asked
    right_keys = ['B{}'.format(idx) for idx in range(right_n_keys)]
    right = right_class(right_keys)

    assert left != right
