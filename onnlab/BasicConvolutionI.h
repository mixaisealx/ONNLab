#pragma once
#include "NNBasicsInterfaces.h"
#include "InputNeuronI.h"

namespace nn::interfaces
{
	class BasicConvolutionEssenceI;

	class BasicConvolutionI {
		friend class BasicConvolutionEssenceI;

		virtual void BatchSizeUpdatedNotify(unsigned new_size) = 0;
	protected:
		void BCE_AddBatchUpdateSubscriber(BasicConvolutionEssenceI *frame, BasicConvolutionI *subscriber);
		void BCE_RemoveBatchUpdateSubscriber(BasicConvolutionEssenceI *frame, BasicConvolutionI *subscriber);

	public:
		virtual ~BasicConvolutionI() = default;

		virtual void PerformFullConvolution() = 0;
		virtual void BackPropagateFullConvolution() = 0;
		virtual void BackPropagateFullConvolutionSpecial(bool update_weights, bool cleanup_grads, void* extra) = 0;

		virtual unsigned GetThreadsCount() = 0;
		virtual void SetThreadsCount(unsigned threads_count) = 0;

		virtual unsigned GetRequiredCachesCount(bool for_backprop) = 0;
		virtual void PerformPartialConvolution(unsigned worker_id, std::vector<float> caches[]) = 0;
		virtual void BackPropagatePartialConvolution(unsigned worker_id, std::vector<float> caches[]) = 0;

		virtual void CleanupLocalCaches() = 0;

		virtual unsigned CalcWeightsCount() = 0;
		virtual std::vector<float> RetrieveWeights() = 0;
		virtual void PushWeights(const std::vector<float> &weights) = 0;
	};

	class BasicConvolutionEssenceI {
		friend class BasicConvolutionI;

		virtual void AddBatchUpdateSubscriber(BasicConvolutionI *subscriber) = 0;
		virtual void RemoveBatchUpdateSubscriber(BasicConvolutionI *subscriber) = 0;
	protected:
		void BC_BatchUpdateNotify(BasicConvolutionI *head, unsigned new_size) {
			head->BatchSizeUpdatedNotify(new_size);
		}
	public:
		virtual void RecalcBatchSize() = 0;
		virtual unsigned GetBatchSize() = 0;
		virtual unsigned GetPlacesCount() = 0;
		virtual unsigned GetKernelSize() = 0;
		virtual unsigned GetKernelsCount() = 0;
		virtual nn::interfaces::NBI *GetInputforLocation(unsigned place_id, unsigned weight_id) = 0;
		virtual nn::interfaces::NBI *GetOutputforLocation(unsigned place_id, unsigned kernel_id) = 0;
		virtual std::vector<unsigned> CalcInputShape() = 0;
		virtual std::vector<unsigned> CalcOutputShape() = 0;
	};
}