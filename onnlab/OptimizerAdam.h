#pragma once
#include "OptimizerI.h"

#include <cmath>

namespace nn::optimizers
{
	class Adam : public nn::interfaces::OptimizerI {
		float learning_rate, beta1, beta2, epsilon;

		static inline double FastDegree(double base, unsigned degree) {
			double result = 1.0f;
			while (degree) {
				if (degree & 1) {
					result *= base;
				}
				base *= base;
				degree >>= 1;
			}
			return result;
		}
	public:
		struct State { // Optimizer state
			float moment1, moment2; 
			unsigned expDecayDegree;
		};

		Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f):learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

		void Reset(void *state) override {
			reinterpret_cast<State *>(state)->moment1 = reinterpret_cast<State *>(state)->moment2 = 0.0f;
			reinterpret_cast<State *>(state)->expDecayDegree = 0;
		}

		float CalcDelta(float gradient, void *state) override {
			State &stt = *reinterpret_cast<State *>(state);

			++stt.expDecayDegree;
			stt.moment1 = stt.moment1 * beta1 + (1.0f - beta1) * gradient;
			stt.moment2 = stt.moment2 * beta2 + (1.0f - beta2) * gradient * gradient;

			double optimizer_moment1_norm = stt.moment1 / (1.0 - FastDegree(beta1, stt.expDecayDegree));
			double optimizer_moment2_norm = stt.moment2 / (1.0 - FastDegree(beta2, stt.expDecayDegree));
			float delta_mod = static_cast<float>(optimizer_moment1_norm / (std::sqrt(optimizer_moment2_norm) + epsilon));

			return learning_rate * delta_mod;
		}

		float GetLearningRate() override {
			return learning_rate;
		}

		void SetLearningRate(float lr = 0.001f) override {
			learning_rate = lr;
		}

		void GetAdamParams(float &beta1, float &beta2, float &epsilon) const {
			beta1 = this->beta1;
			beta2 = this->beta2;
			epsilon = this->epsilon;
		}

		void SetAdamParams(float beta1, float beta2, float epsilon) {
			this->beta1 = beta1;
			this->beta2 = beta2;
			this->epsilon = epsilon;
		}
	};
}
