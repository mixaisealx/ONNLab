#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"
#include "InputNeuronI.h"

#include <vector>
#include <unordered_map>

namespace nn::reverse
{
	class ReverseGuiderB2 {
	public:
		ReverseGuiderB2(std::initializer_list<nn::interfaces::BasicLayerInterface *> layers, std::initializer_list<nn::interfaces::InputNeuronI *> inputs):layers(layers), inputs(inputs) {
			target_output = nullptr;
			input_learning_rate = 0.1f;
		}

		void Reset() {
			input_learning_rate = 0.1f;
			storage.clear();
		}

		void SetTargetOutput(const std::vector<float> *outputs) {
			target_output = outputs;
		}

		void InitInput_value(float value = 0.5f, float input_learning_rate = 0.1f) {
			this->input_learning_rate = input_learning_rate;
			for (auto neuron : inputs) {
				neuron->SetOwnLevel(value);
			}
		}

		void DoForward() {
			for (auto layer : layers) {
				for (auto neuron : layer->Neurons()) {
					neuron->UpdateOwnLevel();
				}
			}
		}

		float BackPropagateError() {
			auto oiter = target_output->begin();
			float mse = 0;
			for (auto opn = layers.back()->Neurons().begin(), eopn = layers.back()->Neurons().end(); opn != eopn; ++opn, ++oiter) {
				float error = (*opn)->OwnLevel() - *oiter;
				storage[*opn].acc_error = error;
				mse += error * error;
			}
			mse /= layers.back()->Neurons().size();

			float ecstep;
			for (auto citer = layers.rbegin(), eiter = layers.rend(); citer != eiter; ++citer) {
				for (auto neuron : (*citer)->Neurons()) {
					ecstep = storage[neuron].acc_error * neuron->ActivationFunctionDerivative(neuron->OwnLevel());
					storage[neuron].acc_error = 0;

					for (auto inpconn : neuron->InputConnections()) {
						storage[inpconn->From()].acc_error += ecstep * inpconn->Weight();
					}
				}
			}
			return mse;
		}

		void OptimizeInput() {
			for (auto neuron : inputs) {
				neuron->SetOwnLevel(neuron->OwnLevel() - input_learning_rate * storage[dynamic_cast<nn::interfaces::NBI *>(neuron)].acc_error);
			}
		}

	private:
		struct MetaData {
			float acc_error;
		};

		std::unordered_map<const void *, MetaData> storage;

		std::vector<interfaces::BasicLayerInterface *> layers;
		std::vector<nn::interfaces::InputNeuronI *> inputs;
		const std::vector<float> *target_output;
		float input_learning_rate;
	};
}
