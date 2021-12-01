from python import *

def test_0():
    desc_0 = make_naive_tensor_descriptor_packed([8, 6, 1, 4])
    desc_1 = make_transform_tensor_descriptor(desc_0,
                    make_tuple(
                        trans_passthrough(desc_0.get_length(0)),
                        trans_unmerge([desc_0.get_length(1) // 3, desc_0.get_length(1) // 2]),
                        trans_passthrough(desc_0.get_length(2)),
                        trans_passthrough(desc_0.get_length(3))
                        ),
                    make_tuple(0, 1, 2, 3),
                    make_tuple(0, [1, 4], 2, 3))


    grouped_slice_lengths = tensor_util_split_lengths(4, desc_1.get_lengths(),
                                            tensor_util_arithmetic_sequence_gen(0, desc_1.get_dims(), 1))
    print(f'desc_1:{desc_1.get_lengths()}')
    desc_2 = make_transform_tensor_descriptor(desc_1,
                    make_tuple(
                        trans_grouped_slice(desc_1.get_lengths(),
                                            tensor_util_uniform_sequence_gen(desc_1.get_dims(), 0),
                                            grouped_slice_lengths)
                        ),
                    make_tuple([0, 1, 2, 3, 4]),
                    make_tuple([0, 1, 2, 3, 4]))

    print(f'desc_2:{desc_2.get_lengths()}')

    desc_3 = make_transform_tensor_descriptor(desc_2,
                    make_tuple(
                        trans_merge([desc_2.get_length(0), desc_2.get_length(2), desc_2.get_length(3)]),
                        trans_passthrough(desc_2.get_length(1)),
                        trans_passthrough(desc_2.get_length(4))
                        ),
                    make_tuple([0, 2, 3], 1, 4),
                    make_tuple(0, 1, 2))

    print(f'desc_3:{desc_3.get_lengths()}')

    coord = [3, 1, 2]
    print(f'coord:{coord}, offset:{desc_3.calculate_offset(coord)}')


if __name__ == '__main__':
    test_0()
 