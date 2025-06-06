#pragma once
#include "NNB_Input.h"
#include "BasicBackPropgI.h"

namespace nn
{
	template<unsigned BATCH_SIZE, bool KahanErrorSummation = false> requires (BATCH_SIZE > 0)
	class NNB_Input_spyableB : public NNB_InputB<BATCH_SIZE>, public interfaces::BasicBackPropogableInterface {
		NNB_Input_spyableB(const NNB_Input_spyableB &) = delete;
		NNB_Input_spyableB &operator=(const NNB_Input_spyableB &) = delete;
	public:
		std::array<float, BATCH_SIZE> backprop_error_accumulator;
		[[msvc::no_unique_address]] [[no_unique_address]] std::conditional<KahanErrorSummation, std::array<float, BATCH_SIZE>, std::monostate>::type backprop_error_accumulator_kahan_compensation;

		NNB_Input_spyableB(float values_storage_array[], unsigned count = 1):NNB_InputB<BATCH_SIZE>(values_storage_array, count) {
			std::fill_n(backprop_error_accumulator.begin(), count, 0.0f);
			if constexpr (KahanErrorSummation) {
				std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), count, 0.0f);
			}
		}
		NNB_Input_spyableB(std::initializer_list<float *> values_sources):NNB_InputB<BATCH_SIZE>(values_sources) {
			std::fill_n(backprop_error_accumulator.begin(), this->GetCurrentBatchSize(), 0.0f);
			if constexpr (KahanErrorSummation) {
				std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), this->GetCurrentBatchSize(), 0.0f);
			}
		}

		NNB_Input_spyableB(std::function<void(float **storage, unsigned capacity, unsigned &used_capacity)> initializer):NNB_InputB<BATCH_SIZE>(initializer) {
			std::fill_n(backprop_error_accumulator.begin(), this->GetCurrentBatchSize(), 0.0f);
			if constexpr (KahanErrorSummation) {
				std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), this->GetCurrentBatchSize(), 0.0f);
			}
		}

		bool IsTrainable() override {
			return true;
		}

		void BackPropResetError() {
			std::fill_n(backprop_error_accumulator.begin(), this->GetCurrentBatchSize(), 0.0f);
			if constexpr (KahanErrorSummation) {
				std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), this->GetCurrentBatchSize(), 0.0f);
			}
		}

		void BackPropAccumulateError(float error, unsigned channel = 0) {
			if constexpr (KahanErrorSummation) {
				float &sum = backprop_error_accumulator[channel];
				float &compensation = backprop_error_accumulator_kahan_compensation[channel];
				float y = error - compensation;
				float t = sum + y;
				compensation = (t - sum) - y;
				sum = t;
			} else {
				backprop_error_accumulator[channel] += error;
			}
		}

		float BackPropGetFinalError(unsigned channel = 0) {
			return backprop_error_accumulator[channel];
		}

		void SetCurrentBatchSize(unsigned batch_size) override {
			if (batch_size && batch_size <= BATCH_SIZE) {
				NNB_InputB<BATCH_SIZE>::SetCurrentBatchSize(batch_size);
				std::fill_n(backprop_error_accumulator.begin(), this->GetCurrentBatchSize(), 0.0f);
				if constexpr (KahanErrorSummation) {
					std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), this->GetCurrentBatchSize(), 0.0f);
				}
			} else
				throw std::exception("batch_size cannot be zero or greater than BATCH_SIZE!");
		}
	};

	using NNB_Input_spyable = NNB_Input_spyableB<1, false>;
}
