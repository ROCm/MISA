#ifndef SEQUENCE_HPP
#define SEQUENCE_HPP

template <int... Is>
struct sequence
{
    static constexpr int m_size = sizeof...(Is);

    __host__ __device__ static constexpr auto size() { return m_size; }

    __host__ __device__ static constexpr auto get_size() { return size(); }

    __host__ __device__ static constexpr int at(int I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const int m_data[m_size + 1] = {Is..., 0};
        return m_data[I];
    }

};
#endif