#pragma once
#include "NNBasicsInterfaces.h"
#include "BatchNeuronBasicI.h"

namespace nn
{
	template <float value>
	class NNB_ConstInputV final : public interfaces::NeuronBasicInterface, public interfaces::BatchNeuronBasicI {
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

		NNB_ConstInputV(const NNB_ConstInputV &) = delete;
		NNB_ConstInputV &operator=(const NNB_ConstInputV &) = delete;
	public:
		NNB_ConstInputV() = default;

		~NNB_ConstInputV() override {
			for (auto out : outputs) {
				out->~ConnectionBasicInterface();
			}
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &OutputConnections() override {
			return outputs;
		}

		void UpdateOwnLevel() override {}

		float OwnLevel(unsigned _ = 0) override {
			return value;
		}

		bool IsTrainable() override {
			return false;
		}

		unsigned GetMaxBatchSize() override {
			return std::numeric_limits<unsigned>::max();
		}

		unsigned GetCurrentBatchSize() override {
			return std::numeric_limits<unsigned>::max();
		}

		void SetCurrentBatchSize(unsigned batch_size) override {}
	};

	using NNB_ConstInput = NNB_ConstInputV<1.0f>;
}
