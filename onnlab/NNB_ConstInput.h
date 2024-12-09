#pragma once
#include "NNBasicsInterfaces.h"

namespace nn
{
	class NNB_ConstInput final : public interfaces::NeuronBasicInterface {
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

		float ActivationFunction(float) const override {
			return 1.0f;
		}

		float ActivationFunctionDerivative(float) const override {
			return 0.0f;
		}

		void AddOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.push_back(output);
		}

		void RemoveOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.erase(std::remove(outputs.begin(), outputs.end(), output), outputs.end());
		}

		NNB_ConstInput(const NNB_ConstInput &) = delete;
		NNB_ConstInput &operator=(const NNB_ConstInput &) = delete;
	public:
		NNB_ConstInput() = default;

		~NNB_ConstInput() override {
			for (auto out : outputs) {
				out->~ConnectionBasicInterface();
			}
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &OutputConnections() override {
			return outputs;
		}

		void UpdateOwnLevel() override {}

		float OwnLevel() override {
			return 1.0f;
		}

		bool IsTrainable() override {
			return false;
		}
	};
}
