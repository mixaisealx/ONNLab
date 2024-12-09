#pragma once

namespace nn::interfaces
{
	class SelectableInputInterface {
	public:
		virtual bool IsBackPropEnabled() = 0;
		virtual void SetBackPropEnabledState(bool is_enabled) = 0;
		virtual void Accumulator_make_NaN() = 0;
		virtual void Accumulator_make_unNaN() = 0;
		virtual float Accumulator_NaN_distance() = 0;
		virtual float Accumulator_unNaN_distance() = 0;
	};
}