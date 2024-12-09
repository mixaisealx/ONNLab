#pragma once
#include "NNBasicsInterfaces.h"

namespace nn::interfaces
{
	class BasicLayerInterface {
	public:
		BasicLayerInterface() = default;
		virtual ~BasicLayerInterface() = default;

		virtual const std::vector<NeuronBasicInterface *> &Neurons() = 0;

		virtual bool HasTrainable() = 0;

		virtual void AddNeuron(NeuronBasicInterface *) = 0;
	};
}
