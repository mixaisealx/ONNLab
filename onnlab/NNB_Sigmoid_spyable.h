#pragma once
#include "NNB_Sigmoid.h"

namespace nn
{
	class NNB_Sigmoid_spyable : public NNB_Sigmoid {
		NNB_Sigmoid_spyable(const NNB_Sigmoid_spyable &) = delete;
		NNB_Sigmoid_spyable &operator=(const NNB_Sigmoid_spyable &) = delete;

		bool do_reset = false;
	public:
		NNB_Sigmoid_spyable() {
		}

		std::vector<float> error_accumulator_archive;

		void BackPropResetError() override {
			NNB_Sigmoid::BackPropResetError();
			do_reset = true;
		}

		void BackPropAccumulateError(float error) override {
			NNB_Sigmoid::BackPropAccumulateError(error);
			if (do_reset) {
				do_reset = false;
				error_accumulator_archive.clear();
			}
			error_accumulator_archive.push_back(error);
		}
	};
}
