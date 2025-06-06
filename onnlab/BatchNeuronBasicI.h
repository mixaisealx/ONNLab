#pragma once

namespace nn::interfaces
{
	class BatchNeuronBasicI {
	public:
		virtual unsigned GetMaxBatchSize() = 0;
		virtual unsigned GetCurrentBatchSize() = 0;
		virtual void SetCurrentBatchSize(unsigned batch_size) = 0;
	};
}
