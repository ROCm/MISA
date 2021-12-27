from python import *

def test_0():
    desc_0 = make_naive_tensor_descriptor_packed([2, 6, 1, 4])
    print(f'desc_0:{desc_0.get_lengths()}')
    desc_1 = make_transform_tensor_descriptor(desc_0,
                    make_tuple(
                        trans_passthrough(desc_0.get_length(0)),
                        trans_unmerge([desc_0.get_length(1) // 2, desc_0.get_length(1) // 3]),
                        trans_passthrough(desc_0.get_length(2)),
                        trans_passthrough(desc_0.get_length(3))
                        ),
                    make_tuple(0, 1, 2, 3),
                    make_tuple(0, [1, 4], 2, 3))

    print(f'desc_1:{desc_1.get_lengths()}')
    assert desc_1.get_lengths() == [2, 3, 1, 4, 2]

    grouped_slice_lengths = tensor_util_split_lengths(4, desc_1.get_lengths(),
                                            tensor_util_arithmetic_sequence_gen(0, desc_1.get_dims(), 1))
    start_coord = tensor_util_uniform_sequence_gen(desc_1.get_dims(), 0)

    start_coord = move_tensor_coordinate(desc_1, start_coord, grouped_slice_lengths)
    print(f'  start_coord:{start_coord}')

    desc_2 = make_transform_tensor_descriptor(desc_1,
                    make_tuple(
                        trans_grouped_slice(desc_1.get_lengths(),
                                            start_coord,
                                            grouped_slice_lengths)
                        ),
                    make_tuple([0, 1, 2, 3, 4]),
                    make_tuple([0, 1, 2, 3, 4]))

    print(f'desc_2:{desc_2.get_lengths()}')
    assert desc_2.get_lengths() == [1, 3, 1, 2, 2]

    desc_3 = make_transform_tensor_descriptor(desc_2,
                    make_tuple(
                        trans_merge([desc_2.get_length(0), desc_2.get_length(1), desc_2.get_length(3)]),
                        trans_passthrough(desc_2.get_length(2)),
                        trans_passthrough(desc_2.get_length(4))
                        ),
                    make_tuple([0, 1, 3], 2, 4),
                    make_tuple(0, 1, 2))

    print(f'desc_3:{desc_3.get_lengths()}')
    assert desc_3.get_lengths() == [6, 1, 2]

    coord = [5, 0, 1]
    print(f'coord:{coord}, offset:{desc_3.calculate_offset(coord)}')
    assert desc_3.calculate_offset(coord) == 23


def test_1():
    vgpr_lengths = list([2, 2, 4, 8])
    vgpr_desc = make_naive_tensor_descriptor_packed(vgpr_lengths)

    vgpr_split_lengths = tensor_util_split_lengths(4, vgpr_lengths, [0, 1, 2, 3], [1, 0, 0, 1])

    start_coord = [0, 0, 0, 0]
    start_coord = move_tensor_coordinate(vgpr_desc, start_coord, vgpr_split_lengths)
    start_coord = move_tensor_coordinate(vgpr_desc, start_coord, vgpr_split_lengths)
    start_coord = move_tensor_coordinate(vgpr_desc, start_coord, vgpr_split_lengths)

    vgpr_split_desc = make_transform_tensor_descriptor(vgpr_desc, 
                                    make_tuple(trans_grouped_slice(vgpr_desc.get_lengths(),
                                                start_coord,
                                                vgpr_split_lengths)),
                                    make_tuple([0, 1, 2, 3]),
                                    make_tuple([0, 1, 2, 3]))

    print(f'lengths:{vgpr_lengths} -> {vgpr_split_lengths}, start coord:{start_coord}')
    assert vgpr_split_lengths == [1, 2, 4, 4]

    sst_vec = 2
    vgpr_co_desc = make_transform_tensor_descriptor(vgpr_split_desc, 
                                    make_tuple(
                                            trans_passthrough(vgpr_split_desc.get_length(0)),
                                            trans_passthrough(vgpr_split_desc.get_length(1)),
                                            trans_passthrough(vgpr_split_desc.get_length(2)),
                                            trans_vectorize(vgpr_split_desc.get_length(3), sst_vec),
                                        ),
                                    make_tuple(0, 1, 2, 3),
                                    make_tuple(0, 1, 2, 3))

    for vgpr_coord in itertools.product(*[range(d) for d in vgpr_co_desc.get_lengths()]):
        vgpr_index = vgpr_co_desc.calculate_offset(list(vgpr_coord))
        print(f'coord:{list(vgpr_coord)}, offset:{vgpr_index}')

if __name__ == '__main__':
    test_0()
    test_1()
 