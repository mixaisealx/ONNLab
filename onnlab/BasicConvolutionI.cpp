#include "BasicConvolutionI.h"

namespace nn::interfaces
{
	void BasicConvolutionI::BCE_AddBatchUpdateSubscriber(BasicConvolutionEssenceI *frame, BasicConvolutionI *subscriber) {
		frame->AddBatchUpdateSubscriber(subscriber);
	}

	void BasicConvolutionI::BCE_RemoveBatchUpdateSubscriber(BasicConvolutionEssenceI *frame, BasicConvolutionI *subscriber) {
		frame->RemoveBatchUpdateSubscriber(subscriber);
	}
}