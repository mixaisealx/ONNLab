#pragma once

namespace nn::interfaces
{
	class BiasedNeuronInterface {
	public:
		virtual float BiasWeight() = 0;
		virtual void BiasWeight(float weight) = 0;
	};
}
