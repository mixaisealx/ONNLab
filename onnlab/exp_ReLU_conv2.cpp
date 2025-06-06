#include "onnlab.h"

#include "NNB_Connection.h"
#include "HyperConnection.h"
#include "OptimizerAdam.h"
#include "NNB_ReLU.h"
#include "NNB_Input.h"
#include "NNB_Storage.h"
#include "NNB_ConstInput.h"
#include "NNB_Layer.h"
#include "NNB_ConvolutionHead.h"
#include "NNB_ConvolutionEssence1d.h"
#include "LearnGuiderFwBPg.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>

#include <iostream>


void exp_ReLU_conv2() {
	std::cout << "exp_ReLU_conv2" << std::endl;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

	// Input vector
	std::array<float, 10> inputs_store;

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input
	nn::NeuronHoldingStaticLayer<nn::NNB_Input> layer_inp(10, [&](nn::NNB_Input *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Input(&inputs_store[index]);
	});

	// Input from convolution
	nn::NeuronHoldingStaticLayer<nn::NNB_Storage> layer_inp2(8, [&](nn::NNB_Storage *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Storage;
	});

	//Hidden
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_hidden(16, [&](nn::NNB_ReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLU;
	});

	// Output
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_out(2, [&](nn::NNB_ReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLU;
	});

	const float LEARNING_RATE = 0.1f;

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);

	// Connections
	// layer_inp -> layer_inp2
	nn::NNB_ConvolutionEssence1d conv1d_frame(&layer_inp, &layer_inp2, 3);
	nn::NNB_ConvolutionHead<OptimAlg> layer_conv1d(&conv1d_frame, &optimizer, [&](bool) { return randistributor(preudorandom); }, false);

	// Hidden
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_inp2_hid(&layer_inp2, &layer_hidden, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// To out
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_hid_out(&layer_hidden, &layer_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	std::uniform_int_distribution<int> noisegen(0, 255);
	std::uniform_int_distribution<unsigned> posegen(0, 7);
	
	// Test generator
	auto GenTest = [&](bool right) {
		// Note: explicit to float/int cast due to suppress annoying warnings
		int avg = 0, max = 0;
		for (auto& elm : inputs_store) {
			elm = (float)noisegen(preudorandom);
			avg += (int)elm;
			if (elm > max) max = (int)elm;
		}
		avg /= inputs_store.size();

		// Removing right patterns
		for (size_t i = 0, lm = inputs_store.size() - 3; i < lm; ++i) {
			int a = (int)(inputs_store[i + 1] - inputs_store[i] * 2);
			int b = (int)(inputs_store[i + 1] - inputs_store[i + 2] * 2);

			if (a > 24 && b > 24) {
				int max2 = (int)std::max(inputs_store[i], inputs_store[i + 2]) + 1;
				int min2 = (int)std::min(inputs_store[i], inputs_store[i + 2]);
				inputs_store[i + 1] = (float)(min2 + noisegen(preudorandom) % (max2 - min2));
			}
		}

		if (right) {
			int pos = posegen(preudorandom);
			if (avg < 96) {
				avg = 96;
				max = 160;
			}
			inputs_store[pos + 1] = (float)(avg + noisegen(preudorandom) % (max - avg));
			inputs_store[pos + 2] = inputs_store[pos] = inputs_store[pos + 1] / 4;
			inputs_store[pos] += noisegen(preudorandom) % 24 - 12;
			inputs_store[pos + 2] += noisegen(preudorandom) % 24 - 12;
		}

		for (auto &elm : inputs_store) {
			elm /= 255;
		}
	};

	// Input to
	nn::errcalc::ErrorCalcSoftMAX softmax_calculator(2);
	nn::LearnGuiderFwBPg learnguider({ &layer_conv1d, &layer_hidden, &layer_out }, &softmax_calculator);

	std::uniform_int_distribution<unsigned> testselector(0, 1);

	std::vector<float> perfect_out_store(2);
	for (size_t iterations = 0; iterations < 11000; ++iterations) {
		// Select datarow
		bool right = testselector(preudorandom) > 0;
		perfect_out_store[0] = !right;
		perfect_out_store[1] = right;

		// Update inputs
		GenTest(right);

		// Learning
		learnguider.DoForward();
		learnguider.FillupOutsError(perfect_out_store);
		learnguider.DoBackward();
	}

	// Accuracy calc
	size_t iters = 10000;
	int rcount = 0;
	for (size_t iterations = 0; iterations < iters; ++iterations) {
		// Select datarow
		bool right = testselector(preudorandom) == 0;

		// Update inputs
		GenTest(right);
		learnguider.DoForward();
		rcount += right == (learnguider.GetOutputs()[0]->OwnLevel() < learnguider.GetOutputs()[1]->OwnLevel());
	}

	std::cout << "Accuracy: " << rcount / (float)iters * 100 << "%\n";
	return;
}