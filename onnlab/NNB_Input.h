#pragma once
#include "NNBasicsInterfaces.h"
#include "InputNeuronI.h"

namespace nn
{
	class NNB_Input final : public interfaces::NeuronBasicInterface, public interfaces::InputNeuronI {
		float *value_storage;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;

		void AddInputConnection(interfaces::ConnectionBasicInterface *) override {
			throw std::exception("Logic error! InputConnection on Input neuron!");
		}

		void RemoveInputConnection(interfaces::ConnectionBasicInterface *) override {
			throw std::exception("Logic error! InputConnection on Input neuron!");
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &InputConnections() override {
			throw std::exception("Logic error! InputConnection on Input neuron!");
		}

		float ActivationFunction(float x) const override {
			return x;
		}

		float ActivationFunctionDerivative(float x) const override {
			return 1;
		}

		void AddOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.push_back(output);
		}

		void RemoveOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.erase(std::remove(outputs.begin(), outputs.end(), output), outputs.end());
		}

		NNB_Input(const NNB_Input &) = delete;
		NNB_Input &operator=(const NNB_Input &) = delete;
	public:
		NNB_Input(float *value_storage):value_storage(value_storage) {
		}

		~NNB_Input() override {
			for (auto out : outputs) {
				out->~ConnectionBasicInterface();
			}
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &OutputConnections() override {
			return outputs;
		}

		float* ValueStoragePtr() {
			return value_storage;
		}

		void ValueStoragePtr(float *value_storage) {
			this->value_storage = value_storage;
		}

		void UpdateOwnLevel() override {
		}

		void SetOwnLevel(float value) override {
			*value_storage = value;
		}

		float OwnLevel() override {
			return *value_storage;
		}

		bool IsTrainable() override {
			return false;
		}
	};
}
