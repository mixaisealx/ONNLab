#include "onnlab.h"

#include "NNB_Connection.h"
#include "HyperConnection.h"
#include "OptimizerAdam.h"
#include "OptimizerGD.h"
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


void exp_ReLU_conv3() {
	std::cout << "exp_ReLU_conv3" << std::endl;

	const unsigned BATCH_SIZE = 4;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);

	// Input vector
	std::array<std::array<float, 10>, BATCH_SIZE> inputs_store;

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(10, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Input from convolution
	nn::NeuronHoldingStaticLayer<nn::NNB_StorageB<BATCH_SIZE>> layer_inp2(8, [&](nn::NNB_StorageB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_StorageB<BATCH_SIZE>;
	});

	//Hidden
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_hidden(16, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});

	// Output
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_out(2, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});

	const float LEARNING_RATE = 0.2f;

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
	auto GenTest = [&](bool right, unsigned idx) {
		auto &input = inputs_store[idx];
		// Note: explicit to float/int cast due to suppress annoying warnings
		int avg = 0, max = 0;
		for (auto &elm : input) {
			elm = (float)noisegen(preudorandom);
			avg += (int)elm;
			if (elm > max) max = (int)elm;
		}
		avg /= input.size();

		// Removing right patterns
		for (size_t i = 0, lm = input.size() - 3; i < lm; ++i) {
			int a = (int)(input[i + 1] - input[i] * 2);
			int b = (int)(input[i + 1] - input[i + 2] * 2);

			if (a > 24 && b > 24) {
				int max2 = (int)std::max(input[i], input[i + 2]) + 1;
				int min2 = (int)std::min(input[i], input[i + 2]);
				input[i + 1] = (float)(min2 + noisegen(preudorandom) % (max2 - min2));
			}
		}

		if (right) {
			int pos = posegen(preudorandom);
			if (avg < 96) {
				avg = 96;
				max = 160;
			}
			input[pos + 1] = (float)(avg + noisegen(preudorandom) % (max - avg));
			input[pos + 2] = input[pos] = input[pos + 1] / 4;
			input[pos] += noisegen(preudorandom) % 24 - 12;
			input[pos + 2] += noisegen(preudorandom) % 24 - 12;
		}

		for (auto &elm : input) {
			elm /= 255;
		}
	};

	// Input to
	nn::errcalc::ErrorCalcSoftMAX softmax_calculator(2);
	nn::LearnGuiderFwBPg learnguider({ &layer_conv1d, &layer_hidden, &layer_out }, &softmax_calculator, BATCH_SIZE);

	std::uniform_int_distribution<unsigned> testselector(0, 1);

	std::array<std::vector<float>, BATCH_SIZE> perfect_out_store;
	perfect_out_store.fill(std::vector<float>(2));

	size_t OVERALL_ITERATIONS = 5000;
	for (size_t iterations = 0, iters = OVERALL_ITERATIONS / BATCH_SIZE; iterations < iters; ++iterations) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			bool right = testselector(preudorandom) > 0;
			perfect_out_store[batch_i][0] = !right;
			perfect_out_store[batch_i][1] = right;

			GenTest(right, batch_i);
		}

		// Learning
		learnguider.DoForward();
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			learnguider.FillupOutsError(perfect_out_store[batch_i], batch_i);
		}
		learnguider.DoBackward();
	}

	// Accuracy calc
	size_t iters = 1000;
	int rcount = 0;
	for (size_t iterations = 0; iterations < iters; ++iterations) {
		// Select datarow
		bool right = testselector(preudorandom) == 0;

		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			GenTest(right, batch_i);
		}

		learnguider.DoForward();
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			rcount += right == (learnguider.GetOutputs()[0]->OwnLevel(batch_i) < learnguider.GetOutputs()[1]->OwnLevel(batch_i));
		}
	}

	std::cout << "Accuracy: " << rcount / (float)(iters * BATCH_SIZE) * 100 << "%\n";
	return;
}