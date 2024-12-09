#pragma once

#include <vector>
#include <algorithm>

namespace nn::utils
{
	class SimpleBitCounter {
		std::vector<uint64_t> storage;
	public:
		SimpleBitCounter(uint16_t items_count = 1U, bool initial_state = false) {
			storage = std::vector<uint64_t>((items_count >> 6) + static_cast<bool>(items_count & 0x3F), (initial_state ? UINT64_MAX : 0));
		}

		void FillWith(bool initial_state = false) {
			std::fill(storage.begin(), storage.end(), (initial_state ? UINT64_MAX : 0));
		}

		uint16_t Capacity() const {
			return storage.size() << 6;
		}

		uint64_t Word(uint16_t index) const {
			return storage[index];
		}

		void Word(uint16_t index, uint64_t word) {
			storage[index] = word;
		}

		bool operator[] (uint16_t index) const {
			return storage[index >> 6] & (static_cast<uint64_t>(0x1) << (index & 0x3F));
		}

		bool GetBit(uint16_t index) const {
			return this->operator[](index);
		}

		void SetBit(uint16_t index, bool bit) {
			uint64_t value = static_cast<uint64_t>(bit) << (index & 0x3F);
			uint64_t &store = storage[index >> 6];
			store = (store & ~value) | value;
		}

		void PlusOne() {
			//bool carry = true;
			for (auto &item : storage) {
				if (item != UINT64_MAX) {
					++item;
					//carry = false;
					break;
				} else {
					item = 0;
					//carry = true;
				}
			}
		}
	};
}