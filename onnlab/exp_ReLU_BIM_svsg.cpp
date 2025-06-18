#include "onnlab.h"

#include "NNB_Connection_spyable.h"
#include "OptimizerAdam.h"
#include "OptimizerGD.h"
#include "NNB_ReLU.h"
#include "NNB_Input_spyable.h"
#include "NNB_Layer.h"
#include "LearnGuiderFwBPg.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"
#include "IterableAggregation.h"

#include "BasicIterativeMethod.h"

#include <vector>
#include <random>
#include <array>

#include <iostream>
#include <iomanip>

static std::vector<float> Learn_ReLU_svsg2();
static bool Validate_ReLU_svsg2(nn::LearnGuiderFwBPg &learnguider);
static std::vector<float> Infer_ReLU_svsg2(nn::LearnGuiderFwBPg &inferguider, std::vector<float> &inputs);

void exp_ReLU_BIM_svsg() {
	std::cout << "exp_ReLU_BIM_svsg" << std::endl;

	auto weights = Learn_ReLU_svsg2();

	const unsigned THREADS_COUNT = 8;

	const unsigned ATTACK_BATCH_SIZE = 8;
	const unsigned RANDOM_SEED = 43;

	// Input vector
	std::array<std::array<float, 7>, ATTACK_BATCH_SIZE> inputs_store;
	for (auto &arr : inputs_store) {
		arr.fill(0.0f);
	}

	// Input layer
	using InputN = nn::NNB_Input_spyableB<ATTACK_BATCH_SIZE, true>;
	nn::NeuronHoldingStaticLayer<InputN> layer_inp(7, [&](InputN *const mem_ptr, unsigned index) {
		new(mem_ptr)InputN([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});
	// Hidden layer 1
	using HiddenN = nn::NNB_ReLUb<ATTACK_BATCH_SIZE, true>;
	nn::NeuronHoldingStaticLayer<HiddenN> layer_relu1(6, [&](HiddenN *const mem_ptr, unsigned) {
		new(mem_ptr)HiddenN;
	});
	// Output layer
	nn::NeuronHoldingStaticLayer<HiddenN> layer_out(11, [&](HiddenN *const mem_ptr, unsigned) {
		new(mem_ptr)HiddenN;
	});
	// Connections
	using NoOptim = nn::optimizers::GradientDescendent;
	NoOptim optimDummy(0.0f); // Disable learning
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>> connections_inp_lr1(&layer_inp, &layer_relu1, [&](nn::NNB_Connection<NoOptim> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<NoOptim>(from, to, &optimDummy);
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>> connections_lr1_out(&layer_relu1, &layer_out, [&](nn::NNB_Connection<NoOptim> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<NoOptim>(from, to, &optimDummy);
	});

	{ // Restoring weights
		using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>>::CBIterator;
		IterableAggregation<nn::interfaces::CBI *> weights_iter;
		weights_iter.AppendIterableItem<IterType>(connections_inp_lr1);
		weights_iter.AppendIterableItem<IterType>(connections_lr1_out);

		std::reverse(weights.begin(), weights.end());
		for (auto ifc : weights_iter) {
			ifc->Weight(weights.back());
			weights.pop_back();
		}
	}

	nn::errcalc::ErrorCalcSoftMAX softmax_calculator(11);
	nn::LearnGuiderFwBPg inferguider({ &layer_inp, &layer_relu1, &layer_out }, &softmax_calculator, ATTACK_BATCH_SIZE);

	if (Validate_ReLU_svsg2(inferguider)) {
		std::cout << "Ready.\n";
	} else {
		std::cout << "Error!\n";
		return;
	}

	std::mt19937 preudorandom(RANDOM_SEED);
	std::uniform_real_distribution<float> randistributor(0.0f, 1.0f);
	std::uniform_int_distribution<unsigned short> randistributor_cls(0, 9);

	std::vector<std::vector<float>> sources(ATTACK_BATCH_SIZE, std::vector<float>(7));
	for (auto &bth : sources) {
		for (auto &pxl : bth) {
			pxl = randistributor(preudorandom);
		}
	}
	std::vector<unsigned short> targets(ATTACK_BATCH_SIZE);
	for (auto &tgt : targets) {
		tgt = randistributor_cls(preudorandom);
	}

	{
		std::cout << "Running BIM...\n";
		nn::reverse::BasicIterativeMethodParams bimparams;
		bimparams.box_min = 0.0f;
		bimparams.box_max = 1.0f;
		bimparams.early_stop_chances_count = 16;
		//bimparams.allow_early_stop = false;

		nn::reverse::BasicIterativeMethod bim(inferguider, bimparams, ATTACK_BATCH_SIZE);

		std::vector<std::vector<float>> sources_copy(sources);
		auto res = bim.RunAttack(sources_copy, targets);

		std::cout << "Results:\n";

		std::cout << std::fixed << std::setprecision(3);

		auto rit = res.begin();
		auto tit = targets.begin();
		for (auto &attk : sources_copy) {
			auto result = Infer_ReLU_svsg2(inferguider, attk);
			unsigned cls = nn::netquality::VectorArgmax(result);

			if (cls == *tit) {
				std::cout << "Success, class: " << cls << " ;" << *rit << "\n";
			} else {
				std::cout << "[!] Failed, target class: " << *tit << " ;" << *rit << "\n";
			}
			++rit;
			++tit;
		}
	}

	{
		std::cout << "========\n";
		std::cout << "Running Multi-Thread BIM...\n";
		nn::LearnGuiderFwBPgThreadAble inferguider_thr({ &layer_inp, &layer_relu1, &layer_out }, ATTACK_BATCH_SIZE, THREADS_COUNT);

		nn::reverse::BasicIterativeMethodParams bimparams;
		bimparams.box_min = 0.0f;
		bimparams.box_max = 1.0f;
		bimparams.early_stop_chances_count = 16;
		//bimparams.allow_early_stop = false;

		nn::reverse::BasicIterativeMethodThreadAble cwl2(inferguider_thr, bimparams, ATTACK_BATCH_SIZE, THREADS_COUNT);

		std::vector<std::vector<float>> sources_copy(sources);
		auto res = cwl2.RunAttack(sources_copy, targets);

		std::cout << "Results Multi-Thread:\n";

		std::cout << std::fixed << std::setprecision(3);

		auto rit = res.begin();
		auto tit = targets.begin();
		for (auto &attk : sources_copy) {
			auto result = Infer_ReLU_svsg2(inferguider, attk);
			unsigned cls = nn::netquality::VectorArgmax(result);

			if (cls == *tit) {
				std::cout << "Success, class: " << cls << " ;" << *rit << "\n";
			} else {
				std::cout << "[!] Failed, target class: " << *tit << " ;" << *rit << "\n";
			}
			++rit;
			++tit;
		}
	}

	return;
}

static std::vector<float> Infer_ReLU_svsg2(nn::LearnGuiderFwBPg &inferguider, std::vector<float> &inputs) {
	std::vector<float> outputs(11);

	auto &inputsnr = inferguider.GetLayers()[0]->Neurons();

	// Update inputs
	for (size_t idx = 0; idx != inputs.size(); ++idx) {
		dynamic_cast<nn::interfaces::InputNeuronI *>(inputsnr[idx])->SetOwnLevel(inputs[idx]);
	}

	inferguider.DoForward();

	for (int i = 0, cnt = inferguider.GetOutputs().size(); i != cnt; ++i) {
		outputs[i] = inferguider.GetOutputs()[i]->OwnLevel();
	}

	return outputs;
}

static bool Validate_ReLU_svsg2(nn::LearnGuiderFwBPg &inferguider) {
	struct datarow {
		datarow(std::initializer_list<unsigned> inputs_nonzero_idx, std::initializer_list<unsigned> outputs_nonzero_idx) {
			inputs.fill(0.0f);
			outputs.fill(0.0f);
			for (auto idx : inputs_nonzero_idx) inputs[idx] = 1.0f;
			for (auto idx : outputs_nonzero_idx) outputs[idx] = 1.0f;
		}
		std::array<float, 7> inputs;
		std::array<float, 11> outputs;
	};
	std::vector<datarow> traindata = {
		datarow({1,2,3,4,5,6}, {0}),
		datarow({2,3}, {1}), datarow({5,6}, {1}),
		datarow({1,2,0,5,4}, {2}),
		datarow({1,2,3,0,4}, {3}),
		datarow({6,0,2,3}, {4}),
		datarow({1,6,0,3,4}, {5}),
		datarow({0,1,3,4,5,6}, {6}),
		datarow({1,2,3}, {7}),
		datarow({0,1,2,3,4,5,6}, {8}),
		datarow({0,1,2,3,4,6}, {9})
	};

	auto &inputs = inferguider.GetLayers()[0]->Neurons();

	std::vector<std::tuple<unsigned, std::tuple<unsigned, float>, std::tuple<unsigned, float>>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		for (size_t idx = 0; idx != sample.inputs.size(); ++idx) {
			dynamic_cast<nn::interfaces::InputNeuronI *>(inputs[idx])->SetOwnLevel(sample.inputs[idx]);
		}
		// Inference
		inferguider.DoForward();

		float max = -std::numeric_limits<float>::infinity();
		int idx = -1;
		for (int i = 0, cnt = inferguider.GetOutputs().size(); i != cnt; ++i) {
			float value = inferguider.GetOutputs()[i]->OwnLevel();
			if (max < value) {
				max = value;
				idx = i;
			}
		}
		int perfect_ans = 0;
		for (int i = 0; i != sample.outputs.size(); ++i) {
			float value = sample.outputs[i];
			if (value > 0.5f) {
				perfect_ans = i;
				break;
			}
		}
		if (perfect_ans != idx) {
			return false;
		}
	}
	return true;
}

static std::vector<float> Learn_ReLU_svsg2() {
	std::vector<float> weights; // For Return-Value-Optimization

	const unsigned BATCH_SIZE = 4;

	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.0f, 1.0f);
	std::uniform_int_distribution<unsigned short> randistributor_int(0, 1);

	// Input vector
	std::array<std::array<float, 7>, BATCH_SIZE> inputs_store;
	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(7, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});
	// Hidden layer 1
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_relu1(6, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});
	// Output layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLUb<BATCH_SIZE>> layer_out(11, [&](nn::NNB_ReLUb<BATCH_SIZE> *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_ReLUb<BATCH_SIZE>;
	});
	// Connections
	using OptimAdam = nn::optimizers::Adam;
	OptimAdam optimizer(0.025f);
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAdam>> connections_inp_lr1(&layer_inp, &layer_relu1, [&](nn::NNB_Connection<OptimAdam> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAdam>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAdam>> connections_lr1_out(&layer_relu1, &layer_out, [&](nn::NNB_Connection<OptimAdam> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAdam>(from, to, &optimizer, randistributor(preudorandom));
	});

	/* Shape is 7-segment digit input.
	*  61112
	*  6   2
	*  00000
	*  5   3
	*  54443
	*/
	struct datarow {
		datarow(std::initializer_list<unsigned> inputs_nonzero_idx, std::initializer_list<unsigned> outputs_nonzero_idx) {
			inputs.fill(0.0f);
			outputs.fill(0.0f);
			for (auto idx : inputs_nonzero_idx) inputs[idx] = 1.0f;
			for (auto idx : outputs_nonzero_idx) outputs[idx] = 1.0f;
		}
		std::array<float, 7> inputs;
		std::array<float, 11> outputs;
	};
	std::vector<datarow> traindata = {
		datarow({1,2,3,4,5,6}, {0}),
		datarow({2,3}, {1}), datarow({5,6}, {1}),
		datarow({1,2,0,5,4}, {2}),
		datarow({1,2,3,0,4}, {3}),
		datarow({6,0,2,3}, {4}),
		datarow({1,6,0,3,4}, {5}),
		datarow({0,1,3,4,5,6}, {6}),
		datarow({1,2,3}, {7}),
		datarow({0,1,2,3,4,5,6}, {8}),
		datarow({0,1,2,3,4,6}, {9})
	};

	datarow wrong_row({}, { 10 });
	auto FillUpWrongRow = [&]() {
		bool is_train;
		do {
			for (auto &itm : wrong_row.inputs) {
				itm = randistributor_int(preudorandom);
			}
			is_train = false;
			for (const auto &row : traindata) {
				if (row.inputs == wrong_row.inputs) {
					is_train = true;
					break;
				}
			}
		} while (is_train);
	};

	nn::errcalc::ErrorCalcSoftMAX softmax_calculator(11);
	nn::LearnGuiderFwBPg learnguider({ &layer_inp, &layer_relu1, &layer_out }, &softmax_calculator, BATCH_SIZE);

	std::uniform_int_distribution<unsigned> testselector(0, traindata.size() - 1);

	std::array<std::vector<float>, BATCH_SIZE> perfect_out_store;
	perfect_out_store.fill(std::vector<float>(11));

	size_t OVERALL_ITERATIONS = 5000;
	for (size_t iterations = 0, iters = OVERALL_ITERATIONS / BATCH_SIZE; iterations < iters; ++iterations) {
		const datarow *row = nullptr;
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			if (randistributor_int(preudorandom)) {
				row = &traindata[testselector(preudorandom)];
			} else {
				FillUpWrongRow();
				row = &wrong_row;
			}
			std::copy(row->inputs.begin(), row->inputs.end(), inputs_store[batch_i].begin());
			std::copy(row->outputs.begin(), row->outputs.end(), perfect_out_store[batch_i].begin());
		}
		learnguider.DoForward();
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
			learnguider.FillupOutsError(perfect_out_store[batch_i], batch_i);
		}
		learnguider.DoBackward();
	}

	using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAdam>>::CBIterator;

	// Storing connections weights
	IterableAggregation<nn::interfaces::CBI *> weights_iter;
	weights_iter.AppendIterableItem<IterType>(connections_inp_lr1);
	weights_iter.AppendIterableItem<IterType>(connections_lr1_out);

	for (auto ifc : weights_iter) {
		weights.push_back(ifc->Weight());
	}
	return weights;
}