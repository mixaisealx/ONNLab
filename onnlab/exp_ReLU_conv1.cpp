#include "onnlab.h"

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
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


void exp_ReLU_conv1() {
	std::cout << "exp_ReLU_conv1" << std::endl;

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
	nn::NeuronHoldingStaticLayer<nn::NNB_Storage> layer_inp2(9, [&](nn::NNB_Storage *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Storage;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_Storage> layer_inp3(3, [&](nn::NNB_Storage *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Storage;
	});

	// Output
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_hd(6, [&](nn::NNB_ReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLU;
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_out(1, [&](nn::NNB_ReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLU;
	});

	const float LEARNING_RATE = 0.1f;

	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD(LEARNING_RATE);

	// Connections
	// layer_inp -> layer_inp2
	nn::NNB_ConvolutionEssence1d conv1d_frame1(&layer_inp, &layer_inp2, 2);
	nn::NNB_ConvolutionHead<OptimGD> layer1_conv1d(&conv1d_frame1, &optimizerGD, [&](bool) { return randistributor(preudorandom); }, false);

	// layer_inp2 -> layer_inp3
	nn::NNB_ConvolutionEssence1d conv1d_frame2(&layer_inp2, &layer_inp3, 3, 3);
	nn::NNB_ConvolutionHead<OptimGD> layer2_conv1d(&conv1d_frame2, &optimizerGD, [&](bool) { return randistributor(preudorandom); }, true);

	// Convolution to out
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_inp3_hd(&layer_inp3, &layer_hd, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_hd_out(&layer_hd, &layer_out, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, randistributor(preudorandom));
	});

	// Input to
	nn::errcalc::ErrorCalcMSE mse_calculator(1);
	nn::LearnGuiderFwBPg learnguider({ &layer1_conv1d, &layer2_conv1d, &layer_hd, &layer_out }, &mse_calculator);

	std::uniform_int_distribution<unsigned> testselector(0, 1);

	std::uniform_int_distribution<int> noisegen(1, 5);
	std::uniform_int_distribution<unsigned> posegen(0, 8);

	auto GenTest = [&](bool right) {
		// Note: explicit to float/int cast due to suppress annoying warnings
		for (auto &elm : inputs_store) {
			elm = (float)noisegen(preudorandom);
		}

		// Removing right patterns
		for (size_t i = 0, lm = inputs_store.size() - 1; i < lm; ++i) {
			if (inputs_store[i] - inputs_store[i + 1] > 4.5f) {
				inputs_store[i + 1] = 0.5f * noisegen(preudorandom);
			}
		}

		if (right) {
			int pos = posegen(preudorandom);
			inputs_store[pos] = 5.0f;
			inputs_store[pos + 1] = 1.0f;
		}

		for (auto &elm : inputs_store) {
			elm /= 5;
		}
	};

	std::vector<float> perfect_out_store(1);
	for (size_t iterations = 0; iterations < 20000; ++iterations) {
		// Select datarow
		bool right = testselector(preudorandom) == 0;
		perfect_out_store[0] = right;

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
		rcount += right == (learnguider.GetOutputs()[0]->OwnLevel() > 0.5f);
	}

	std::cout << "Accuracy: " << rcount / (float)iters * 100 << "%\n";

	return;
}