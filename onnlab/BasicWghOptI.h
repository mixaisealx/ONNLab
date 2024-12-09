#pragma once

namespace nn::interfaces
{
	class BasicWeightOptimizableInterface {
	public:
		virtual void WeightOptimReset() = 0;
		virtual void WeightOptimDoUpdate(float delta) = 0;
	};
}