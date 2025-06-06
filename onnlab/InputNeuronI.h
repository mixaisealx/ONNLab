#pragma once

namespace nn::interfaces
{
	class InputNeuronI {
	public:
		virtual void SetOwnLevel(float value, unsigned data_channel = 0) = 0;
		virtual float OwnLevel(unsigned data_channel = 0) = 0;
	};
}
