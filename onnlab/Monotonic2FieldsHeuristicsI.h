#pragma once
#include "NNBasicsInterfaces.h"

namespace nn::interfaces
{
	class Monotonic2FieldsHeuristicsI {
	public:
		using ProjectorTwoToOne = void(interfaces::NBI *const non_monotonic, const interfaces::NBI *const equvivalent_1, const interfaces::NBI *const equvivalent_2);
		using ProjectorOneToTwo = void(const interfaces::NBI *const non_monotonic, interfaces::NBI *const equvivalent_1, interfaces::NBI *const equvivalent_2);

		virtual void Perform2to1(interfaces::NBI *const non_monotonic, const interfaces::NBI *const equvivalent_1, const interfaces::NBI *const equvivalent_2, size_t index_payload) = 0;
		virtual void Perform1to2(const interfaces::NBI *const non_monotonic, interfaces::NBI *const equvivalent_1, interfaces::NBI *const equvivalent_2, size_t index_payload) = 0;
	};
}
