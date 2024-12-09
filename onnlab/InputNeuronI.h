#pragma once

namespace nn::interfaces
{
	class InputNeuronI {
	public:
		virtual void SetOwnLevel(float value) = 0;
		virtual float OwnLevel() = 0;
	};
}
