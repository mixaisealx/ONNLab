#include "onnlab.h"

#include "NNB_Connection.h"
#include "OptimizerGD.h"
#include "NNB_ReLU.h"
#include "NNB_Input_spyable.h"
#include "NNB_Storage.h"
#include "NNB_Layer.h"
#include "LearnGuiderFwBPg.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"
#include "NNB_ConvolutionHead.h"
#include "NNB_ConvolutionEssence2d.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <cassert>

#include <iostream>


void exp_stabtest2() {
	std::cout << "exp_stabtest2" << std::endl;

	const unsigned BATCH_SIZE = 8;
	const unsigned SINGLE_CHANNEL = 3;

	const unsigned INPS_COUNT_SQUARE = 8;
	const unsigned KERNEL_SZ = 3;
	const unsigned OUTS_COUNT = 2;


	std::uniform_real_distribution<float> randistributor(-0.5f, 0.5f);

	// Input vector
	std::vector<std::array<float, INPS_COUNT_SQUARE * INPS_COUNT_SQUARE>> inputs_store(BATCH_SIZE);

	// No batch
	nn::NeuronHoldingStaticLayer<nn::NNB_Input_spyable> layer_inp(INPS_COUNT_SQUARE * INPS_COUNT_SQUARE, [&](nn::NNB_Input_spyable *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Input_spyable(&inputs_store[SINGLE_CHANNEL][index]);
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_Storage> layer_inp2((INPS_COUNT_SQUARE - KERNEL_SZ + 1) * (INPS_COUNT_SQUARE - KERNEL_SZ + 1) * 2, [&](nn::NNB_Storage *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Storage;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_relu1(10, [&](nn::NNB_ReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLU;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_out(OUTS_COUNT, [&](nn::NNB_ReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLU;
	});

	// Batch
	nn::NeuronHoldingStaticLayer<nn::NNB_Input_spyableB<BATCH_SIZE>> batch_layer_inp(INPS_COUNT_SQUARE * INPS_COUNT_SQUARE, [&](nn::NNB_Input_spyableB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Input_spyableB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_StorageB<BATCH_SIZE>> batch_layer_inp2((INPS_COUNT_SQUARE - KERNEL_SZ + 1) * (INPS_COUNT_SQUARE - KERNEL_SZ + 1) * 2, [&](nn::NNB_StorageB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_StorageB<BATCH_SIZE>;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> batch_layer_relu1(10, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> batch_layer_out(OUTS_COUNT, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});

	// Connections
	const float LEARNING_RATE = 0.1f;
	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD(LEARNING_RATE);

	// No batch
	nn::NNB_ConvolutionEssence2d c1onv2d_frame(&layer_inp, &layer_inp2, INPS_COUNT_SQUARE, 
											   nn::NNB_ConvolutionEssence2d::KernelShape(KERNEL_SZ), nn::NNB_ConvolutionEssence2d::KernelMovement(), 2);
	nn::NNB_ConvolutionHead<OptimGD> layer_c1onv2d(&c1onv2d_frame, &optimizerGD, [](bool) { return 0.0f; });

	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_inp_lr1(&layer_inp2, &layer_relu1, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD);
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_lr1_out(&layer_relu1, &layer_out, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD);
	});

	// Batch
	nn::NNB_ConvolutionEssence2d batch_conv2d_frame(&batch_layer_inp, &batch_layer_inp2, INPS_COUNT_SQUARE,
													nn::NNB_ConvolutionEssence2d::KernelShape(KERNEL_SZ), nn::NNB_ConvolutionEssence2d::KernelMovement(), 2);
	nn::NNB_ConvolutionHead<OptimGD> batch_layer_conv2d(&batch_conv2d_frame, &optimizerGD, [](bool) { return 0.0f; });

	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> batch_connections_inp_lr1(&batch_layer_inp2, &batch_layer_relu1, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD);
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> batch_connections_lr1_out(&batch_layer_relu1, &batch_layer_out, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD);
	});

	auto InitWeights = [&](nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> &connections, std::mt19937 &randomgen) {
		for (auto &conn : connections.ConnectionsInside()) {
			conn.Weight(randistributor(randomgen));
		}
	};
	auto InitWeightsConvolution = [&](nn::NNB_ConvolutionHead<OptimGD> &conv, std::mt19937 &randomgen) {
		unsigned wcount = conv.CalcWeightsCount();
		std::vector<float> temp;
		temp.reserve(wcount);
		for (size_t i = 0; i < wcount; i++) {
			temp.push_back(randistributor(randomgen));
		}
		conv.PushWeights(temp);
	};

	nn::errcalc::ErrorCalcSoftMAX softmax_calculator1(OUTS_COUNT);
	nn::LearnGuiderFwBPg learnguider1({ &layer_c1onv2d, &layer_relu1, &layer_out }, &softmax_calculator1);

	nn::errcalc::ErrorCalcSoftMAX softmax_calculator2(OUTS_COUNT);
	nn::LearnGuiderFwBPg learnguider2({ &batch_layer_conv2d,&batch_layer_relu1, &batch_layer_out }, &softmax_calculator2, BATCH_SIZE);

	std::array<std::vector<float>, BATCH_SIZE> perfect_out_store;
	perfect_out_store.fill(std::vector<float>(OUTS_COUNT));

	std::mt19937 preudorandom0(37);
	for (auto &row : inputs_store) {
		for (auto &flt : row) {
			flt = randistributor(preudorandom0);
		}
	}
	for (auto &row : perfect_out_store) {
		for (auto &flt : row) {
			flt = randistributor(preudorandom0);
		}
	}

	std::array<float, OUTS_COUNT> prew_outs;
	std::array<float, INPS_COUNT_SQUARE *INPS_COUNT_SQUARE> prew_backprop;
	for (size_t j = 0; j < 5000; j++) {
		std::mt19937 preudorandom1(42);
		InitWeightsConvolution(layer_c1onv2d, preudorandom1);
		InitWeights(connections_inp_lr1, preudorandom1);
		InitWeights(connections_lr1_out, preudorandom1);

		std::mt19937 preudorandom2(42);
		InitWeightsConvolution(batch_layer_conv2d, preudorandom2);
		InitWeights(batch_connections_inp_lr1, preudorandom2);
		InitWeights(batch_connections_lr1_out, preudorandom2);

		assert(randistributor(preudorandom1) == randistributor(preudorandom2));

		// Learning
		learnguider1.DoForward();
		learnguider2.DoForward();
		assert(learnguider1.GetOutputs().size() == learnguider2.GetOutputs().size());
		for (size_t i = 0; i < learnguider1.GetOutputs().size(); i++) {
			assert(learnguider1.GetOutputs()[i]->OwnLevel() == learnguider2.GetOutputs()[i]->OwnLevel(SINGLE_CHANNEL));
			if (j == 0) {
				prew_outs[i] = learnguider1.GetOutputs()[i]->OwnLevel();
			} else {
				assert(learnguider1.GetOutputs()[i]->OwnLevel() == prew_outs[i]);
			}
		}

		float loss1 = learnguider1.FillupOutsError(perfect_out_store[SINGLE_CHANNEL], 0, true);
		float loss2[BATCH_SIZE];
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			loss2[batch_i] = learnguider2.FillupOutsError(perfect_out_store[batch_i], batch_i, true);
		}
		assert(loss1 == loss2[SINGLE_CHANNEL]);

		learnguider1.DoBackward();
		learnguider2.DoBackward();

		auto &inp1 = layer_inp.Neurons();
		auto &inp2 = batch_layer_inp.Neurons();
		assert(inp1.size() == inp2.size());
		for (size_t i = 0; i < inp1.size(); i++) {
			auto pinp1 = dynamic_cast<nn::NNB_Input_spyable *>(inp1[i]);
			auto pinp2 = dynamic_cast<nn::NNB_Input_spyableB<BATCH_SIZE> *>(inp2[i]);
			assert(pinp1->backprop_error_accumulator[0] == pinp2->backprop_error_accumulator[SINGLE_CHANNEL]);
			if (j == 0) {
				prew_backprop[i] = pinp1->backprop_error_accumulator[0];
			} else {
				assert(pinp1->backprop_error_accumulator[0] == prew_backprop[i]);
			}
			pinp1->BackPropResetError();
			pinp2->BackPropResetError();
		}
	}

	std::cout << "OK\n";
	return;
}