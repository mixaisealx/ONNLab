#pragma once

namespace nn::interfaces {
	class BasicBackPropogableInterface {
	public:
		virtual void BackPropResetError() = 0;
		virtual void BackPropAccumulateError(float error) = 0;
		virtual float BackPropGetFinalError() = 0;
	};

	class MaccBackPropogableInterface : public BasicBackPropogableInterface {
	public:
		virtual float OwnAccumulatorValue() = 0;
	};
}