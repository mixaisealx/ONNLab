#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicWghOptI.h"

namespace nn
{
	class NNB_Connection : public interfaces::ConnectionBasicInterface, public interfaces::BasicWeightOptimizableInterface {
		interfaces::NeuronBasicInterface *from;
		interfaces::NeuronBasicInterface *to;
		float weight;
		float optimizer_learning_rate, optimizer_beta1, optimizer_beta2, optimizer_epsilon; // Optimizer settings
		float optimizer_moment1, optimizer_moment2; // Optimizer state
		unsigned optimizer_expDecayDegree; // Optimizer state

		NNB_Connection(const NNB_Connection &) = delete;
		NNB_Connection &operator=(const NNB_Connection &) = delete;

		static inline float FastDegree(float base, unsigned degree) {
			float result = 1.0f;
			while (degree) {
				if (degree & 1) {
					result = result * base;
				}
				base = base * base;
				degree >>= 1;
			}
			return result;
		}
	public:
		// Using Gradient Decedent
		NNB_Connection(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to, float initial_weight = 0.0f, float optimizer_learning_rate = 0.1f): from(from), to(to), weight(initial_weight), optimizer_learning_rate(optimizer_learning_rate){
			NBI_AddOutputConnection(from, this);
			NBI_AddInputConnection(to, this);
			optimizer_beta1 = optimizer_beta2 = optimizer_epsilon = -1.0f;
			WeightOptimReset();
		}

		// Using Adam
		NNB_Connection(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to, bool useAdam, float initial_weight = 0.0f, float optimizer_learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f): 
			from(from), 
			to(to), 
			weight(initial_weight), 
			optimizer_learning_rate(optimizer_learning_rate),
			optimizer_beta1(beta1),
			optimizer_beta2(beta2),
			optimizer_epsilon(epsilon)
		{
			if (!useAdam) throw std::logic_error("Remove \"useAdam\" flag to use GD instead.");
			NBI_AddOutputConnection(from, this);
			NBI_AddInputConnection(to, this);
			WeightOptimReset();
		}

		~NNB_Connection() override {
			if (from && to) {
				NBI_RemoveOutputConnection(from, this);
				NBI_RemoveInputConnection(to, this);
				from = to = nullptr;
			}
		}

		interfaces::NeuronBasicInterface *From() override {
			return from;
		}

		interfaces::NeuronBasicInterface *To() override {
			return to;
		}

		float Weight() override {
			return weight;
		}

		void Weight(float weight) override {
			this->weight = weight;
		}

		void WeightOptimReset() override {
			optimizer_moment1 = optimizer_moment2 = 0.0f;
			optimizer_expDecayDegree = 0;
		}

		void WeightOptimDoUpdate(float delta) override {
			if (optimizer_epsilon > 0) { // Use Adam
				float gradient = delta * from->OwnLevel();

				++optimizer_expDecayDegree;
				optimizer_moment1 = optimizer_moment1 * optimizer_beta1 + (1.0f - optimizer_beta1) * gradient;
				optimizer_moment2 = optimizer_moment2 * optimizer_beta2 + (1.0f - optimizer_beta2) * gradient * gradient;

				float optimizer_moment1_norm = optimizer_moment1 / (1.0f - FastDegree(optimizer_beta1, optimizer_expDecayDegree));
				float optimizer_moment2_norm = optimizer_moment2 / (1.0f - FastDegree(optimizer_beta2, optimizer_expDecayDegree));
				float delta_mod = optimizer_moment1_norm / (std::sqrt(optimizer_moment2_norm) + optimizer_epsilon);

				weight -= optimizer_learning_rate * delta_mod;
			} else { // Use GD
				weight -= optimizer_learning_rate * delta * from->OwnLevel();
			}
		}
	};
}
