#include "onnlab.h"

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "OptimizerGD.h"
#include "NNB_ReLU.h"
#include "NNB_nm1ReLU.h"
#include "NNB_Input.h"
#include "NNB_Layer.h"
#include "NNB_LayersAggregator.h"
#include "NNB_ConstInput.h"
#include "NNB_LinearSlim.h"
#include "LearnGuiderFwBPg.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "SparceLayerStaticConnectomHolder2Mult.h"
#include "NeuronHoldingStaticLayer.h"
#include "Monotonic2FieldsProjectingAccessory.h"
#include "Monotonic2FieldsHeuristicsEqExV1.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <algorithm>

#include <iostream>


void exp_nm1ReLU_svsg3() {
	std::cout << "exp_nm1ReLU_svsg3" << std::endl;

	const float NM1_MAX_VALUE = 3.0f;

	//std::random_device randevice;
	std::mt19937 preudorandom(42);
	std::uniform_real_distribution<float> randistributor(0.2f, 0.4f);
	std::uniform_real_distribution<float> randistributor_small(0.001f, 0.1f);
	std::uniform_int_distribution<unsigned short> randistributor_int(0, 1);

	using nnNmReLU = nn::NNB_nm1ReLUb<1, false, true>;

	// Input vector
	std::array<float, 7> inputs_store;

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_Input> layer_inp(7, [&](nn::NNB_Input *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_Input(&inputs_store[index]);
	});

	// Output m1ReLU layer 
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer_m1out(11, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE, 0.1f);
	});

	// Equvivelent ReLU layer 
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_rleq_inc(11, [&](nn::NNB_ReLU *const mem_ptr, unsigned) { // Increasing (__/)
		new(mem_ptr)nn::NNB_ReLU(0.1f);
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_ReLU> layer_rleq_dec(11, [&](nn::NNB_ReLU *const mem_ptr, unsigned) { // Decreasing (\__)
		new(mem_ptr)nn::NNB_ReLU(0.1f);
	});
	nn::NeuronHoldingStaticLayer<nn::NNB_LinearSlim> layer_smout(11, [&](nn::NNB_LinearSlim *const mem_ptr, unsigned) {
		new(mem_ptr)nn::NNB_LinearSlim;
	});

	nn::NNB_LayersAggregator layer_rleq({ &layer_rleq_inc, &layer_rleq_dec });

	// Connections
	using OptimGD = nn::optimizers::GradientDescendent;
	OptimGD optimizerGD;
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_bias_m1out(&layer_bias, &layer_m1out, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, -NM1_MAX_VALUE);
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_bias_rleqi(&layer_bias, &layer_rleq_inc, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, -randistributor_small(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_bias_rleqd(&layer_bias, &layer_rleq_dec, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, randistributor_small(preudorandom));
	});
	// Input to
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_in_m1out(&layer_inp, &layer_m1out, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_in_rleqi(&layer_inp, &layer_rleq_inc, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimGD>> connections_in_rleqd(&layer_inp, &layer_rleq_dec, [&](nn::NNB_Connection<OptimGD> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimGD>(from, to, &optimizerGD, -randistributor(preudorandom));
	});
	// Equvivelent head
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connections_rleq_smout(&layer_rleq, &layer_smout, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});

	nn::Monotonic2FieldsHeuristicsEqExV1 mon2filds(NM1_MAX_VALUE);
	nn::Monotonic2FieldsProjectingAccessory projector_rl_m1(&layer_m1out, &layer_rleq, &mon2filds);

	// Train data
	/* Shape is 7-segment digit input. Task: recognize digit.
	*  61112
	*  6   2
	*  00000
	*  5   3
	*  54443
	*/
	struct datarow {
		datarow(std::initializer_list<unsigned> inputs_nonzero_idx, std::initializer_list<unsigned> outputs_nonzero_idx) {
			inputs.fill(0.0f); outputs.resize(11, 0.0f);
			for (auto idx : inputs_nonzero_idx) inputs[idx] = 1.0f;
			for (auto idx : outputs_nonzero_idx) outputs[idx] = 1.0f;
		}
		std::array<float, 7> inputs;
		std::vector<float> outputs;
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

	datarow wrong_row({}, {10});

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
	nn::LearnGuiderFwBPg learnguider_rl({ &layer_inp, &layer_rleq, &layer_smout }, &softmax_calculator);

	nn::LearnGuiderFwBPg learnguider_m1({ &layer_inp, &layer_m1out }, &softmax_calculator);

	std::uniform_int_distribution<unsigned> testselector(0, traindata.size() - 1);

	bool stand_learn = true;

	for (size_t iterations = 0; iterations < 3300; ++iterations) {
		// Select datarow
		const datarow *row = nullptr;
		//row = &traindata[testselector(preudorandom)];
		if (randistributor_int(preudorandom)) {
			row = &traindata[testselector(preudorandom)];
		} else {
			FillUpWrongRow();
			row = &wrong_row;
		}

		// Update inputs
		std::copy(row->inputs.begin(), row->inputs.end(), inputs_store.begin());
		const auto &perfect_out = row->outputs;

		// Learning
		if (stand_learn) {
			learnguider_rl.DoForward();
			learnguider_rl.FillupOutsError(perfect_out);
			learnguider_rl.DoBackward();
		} else {
			learnguider_m1.DoForward();
			learnguider_m1.FillupOutsError(perfect_out);
			learnguider_m1.DoBackward();
		}

		if (iterations == 800) {
			projector_rl_m1.Perform2to1LossyCompression();
			stand_learn = false;
		} /*else if (iterations == 2000) {
			projector_rl_m1.Perform1to2DiffTransfer();
			stand_learn = true;
		}*/
	}

	// Non-monotonity usage stats: minus_count, plus_count, min, summ, max
	std::map<const nnNmReLU *, std::tuple<unsigned, unsigned, float, float, float>> fu_both_neurons;

	auto NonMonotonityStatProc = [&](nnNmReLU &nrn) {
		auto iter = fu_both_neurons.find(&nrn);
		if (iter != fu_both_neurons.end()) {
			if (nrn.RealAccumulatorValue() < 0) {
				++std::get<0>(iter->second);
			} else {
				++std::get<1>(iter->second);
			}
			std::get<3>(iter->second) += nrn.RealAccumulatorValue();
			if (nrn.RealAccumulatorValue() < std::get<2>(iter->second)) {
				std::get<2>(iter->second) = nrn.RealAccumulatorValue();
			} else if (nrn.RealAccumulatorValue() > std::get<4>(iter->second)) {
				std::get<4>(iter->second) = nrn.RealAccumulatorValue();
			}
		} else {
			bool minus = nrn.RealAccumulatorValue() < 0;
			fu_both_neurons.emplace(&nrn, std::make_tuple((unsigned)(minus), (unsigned)(!minus), nrn.RealAccumulatorValue(), nrn.RealAccumulatorValue(), nrn.RealAccumulatorValue()));
		}
	};
	// Inferencing
	std::cout << "learnguider_rl" << std::endl;

	for (const auto &sample : traindata) {
		// Update inputs
		std::copy(sample.inputs.begin(), sample.inputs.end(), inputs_store.begin());

		// Inference
		learnguider_rl.DoForward();

		float max = std::numeric_limits<float>::min(), max2 = max;
		unsigned idx = 0, idx2 = 0;
		for (unsigned i = 0, cnt = layer_smout.Neurons().size(); i != cnt; ++i) {
			float value = layer_smout.Neurons()[i]->OwnLevel();
			if (max < value) {
				max2 = max;
				idx2 = idx;
				max = value;
				idx = i;
			} else if (max2 < value) {
				max2 = value;
				idx2 = i;
			}
		}
		unsigned perfect_ans = 0;
		for (unsigned i = 0; i != sample.outputs.size(); ++i) {
			float value = sample.outputs[i];
			if (value > 0.5f) {
				perfect_ans = i;
				break;
			}
		}
		std::cout << perfect_ans << " | " << idx << '\t' << max << " | " << idx2 << '\t' << max2 << std::endl;
	}

	std::cout << std::endl << "learnguider_m1" << std::endl;

	std::vector<std::tuple<unsigned, std::tuple<unsigned, float>, std::tuple<unsigned, float>>> results;
	for (const auto &sample : traindata) {
		// Update inputs
		std::copy(sample.inputs.begin(), sample.inputs.end(), inputs_store.begin());

		// Inference
		learnguider_m1.DoForward();

		float max = std::numeric_limits<float>::min(), max2 = max;
		unsigned idx = 0, idx2 = 0;
		for (unsigned i = 0, cnt = layer_m1out.Neurons().size(); i != cnt; ++i) {
			float value = layer_m1out.Neurons()[i]->OwnLevel();
			if (max < value) {
				max2 = max;
				idx2 = idx;
				max = value;
				idx = i;
			} else if (max2 < value) {
				max2 = value;
				idx2 = i;
			}
		}
		unsigned perfect_ans = 0;
		for (unsigned i = 0; i != sample.outputs.size(); ++i) {
			float value = sample.outputs[i];
			if (value > 0.5f) {
				perfect_ans = i;
				break;
			}
		}
		results.push_back(std::make_tuple(perfect_ans, std::make_tuple(idx, max), std::make_tuple(idx2, max2)));

		// Grab non-monotonity stat
		for (auto &nrn : layer_m1out.NeuronsInside()) {
			NonMonotonityStatProc(nrn);
		}
	}

	for (auto &iter : fu_both_neurons) {
		std::get<3>(iter.second) /= (std::get<0>(iter.second) + std::get<1>(iter.second)) * NM1_MAX_VALUE; // Normalizing average
		std::get<2>(iter.second) /= NM1_MAX_VALUE; // Normalizing miniumm
		std::get<4>(iter.second) /= NM1_MAX_VALUE; // Normalizing maximum
	}

	for (const auto &tpl : results) {
		std::cout << std::get<0>(tpl) << " | " << std::get<0>(std::get<1>(tpl)) << '\t' << std::get<1>(std::get<1>(tpl)) << " | " << std::get<0>(std::get<2>(tpl)) << '\t' << std::get<1>(std::get<2>(tpl)) << std::endl;
	}

	return;
}