#include "onnlab.h"

#include "NNB_Connection.h"
#include "NNB_StraightConnection.h"
#include "OptimizerGD.h"
#include "OptimizerAdam.h"
#include "NNB_Linear.h"
#include "NNB_ReLU.h"
#include "NNB_ilReLU.h"
#include "NNB_nm1iReLU.h"
#include "NNB_nm1ReLU.h"
#include "NNB_Input.h"
#include "NNB_ConstInput.h"
#include "NNB_Input_spyable.h"
#include "NNB_LinearSlim.h"
#include "NNB_Layer.h"
#include "NNB_LayersAggregator.h"
#include "LearnGuiderFwBPgThreadAble.h"
#include "NeuronHoldingStaticLayer.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "SparceLayerStaticConnectomHolder2Mult.h"
#include "Monotonic2FieldsProjectingAccessory.h"
#include "Monotonic2FieldsHeuristicsEqExV2.h"

#include "CSVreader.h"
#include "IterableAggregation.h"
#include "NetQualityCalcUtils.h"

#include "BasicIterativeMethod.h"
#include "CarliniWagnerL2.h"

#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <algorithm>
#include <thread>

#include <iostream>
#include <fstream>
#include <syncstream>
#include <iomanip>


static unsigned THREADS_COUNT = 16;
static const unsigned LOG_INTERVAL = 400;

static const unsigned BATCH_SIZE = 64;
static const unsigned ATTACK_BATCH_SIZE = 64;
static const unsigned ATTACK_REPEAT_COUNT = 2;

static const unsigned EPOCHS = 10;
static const float VALID_TEST_SPLIT_RATIO = 0.1f; // sizeof(validation)/sizeof(full testing dataset size), should be small to run fast

static const float LEARNING_RATE = 0.001f;

static const unsigned RELU_LAYER_HIDDEN1_SIZE = 32;
static const unsigned NM_LAYER_HIDDEN1_SIZE = 32;

static const unsigned LAYER_HIDDEN2_SIZE = 16;

static const float NM1_MAX_VALUE = 8.0f;
static const float PROJECTION_THRESHOLD = -0.4f;

struct MNIST_entry {
	MNIST_entry(uint8_t label, std::vector<uint8_t> input, uint16_t reseve):label(label), input(input) {
		input.reserve(reseve);
	}
	uint8_t label;
	std::vector<uint8_t> input;
};

const unsigned INPUT_SIZE = 28 * 28;

struct MNIST_Attack_ReLU_result { float accuracy, f1; float cwl2_success_rate, bim_success_rate; unsigned cwl2_iters, bim_iters; };
struct MNIST_Attack_nmiReLU_result { float accuracy, f1; float stand_percent, extra_percent; float cwl2_success_rate, bim_success_rate; unsigned cwl2_iters, bim_iters; };

static inline MNIST_Attack_ReLU_result MNIST_Attack_ReLU(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test);

static inline MNIST_Attack_nmiReLU_result MNIST_Attack_nmiReLU_eq_nmi(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test);
static inline MNIST_Attack_nmiReLU_result MNIST_Attack_nmiReLU_nmi(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test);
static inline MNIST_Attack_nmiReLU_result MNIST_Attack_nmiReLU_nm0(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test);


void exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1() {
	std::cout << "exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1" << std::endl;

	std::cout << "Enter the overall threads count: ";
	{
		unsigned temp;
		std::cin >> temp;
		THREADS_COUNT = temp / 2;
	}
	std::cout << "Resulting threads distribution: " << THREADS_COUNT << " + " << THREADS_COUNT << " = " << THREADS_COUNT + THREADS_COUNT << '\n';

	std::cout << "Loading train dataset...\n";

	std::vector<MNIST_entry> mnist_train;
	{
		CSVreader csv("../mnist-in-csv/mnist_train.csv");

		auto rowsz = csv.FetchNextRow();
		while (rowsz) {
			mnist_train.emplace_back(atoi(csv[0]), std::vector<uint8_t>(), rowsz - 1); // label writing

			for (unsigned i = 1; i < rowsz; ++i) {
				mnist_train.back().input.push_back(atoi(csv[i]));
			}
			rowsz = csv.FetchNextRow();
		}
	}
	nn::netquality::ClassWeightsCalculator mnist_train_classes_w(10);
	for (auto &itm : mnist_train) {
		mnist_train_classes_w.NoteSample(itm.label);
	}
	mnist_train_classes_w.CalcWeights();
	mnist_train_classes_w.KeepWeightsOnly();

	std::cout << "Loading test dataset (and valid/test split)...\n";

	std::vector<MNIST_entry> mnist_test;
	{
		CSVreader csv("../mnist-in-csv/mnist_test.csv");

		auto rowsz = csv.FetchNextRow();
		while (rowsz) {
			mnist_test.emplace_back(atoi(csv[0]), std::vector<uint8_t>(), rowsz - 1); // label writing

			for (unsigned i = 1; i < rowsz; ++i) {
				mnist_test.back().input.push_back(atoi(csv[i]));
			}
			rowsz = csv.FetchNextRow();
		}
	}
	nn::netquality::ClassWeightsCalculator mnist_test_classes_w(10);
	for (auto &itm : mnist_test) {
		mnist_test_classes_w.NoteSample(itm.label);
	}
	mnist_test_classes_w.CalcWeights();
	mnist_test_classes_w.KeepWeightsOnly();

	std::random_device randevice;

	std::mt19937 preudorand_determined(42);
	std::shuffle(mnist_test.begin(), mnist_test.end(), preudorand_determined);
	std::vector<MNIST_entry> mnist_valid;
	mnist_valid.reserve(static_cast<unsigned>(VALID_TEST_SPLIT_RATIO * mnist_test.size()));
	{
		std::copy_n(mnist_test.begin(), mnist_valid.capacity(), std::back_inserter(mnist_valid));
	}

	AtomicSpinlock locker;

	auto ModelRunner_ReLU = [&](unsigned num) {
		locker.lock();
		unsigned random_seed = randevice();
		locker.unlock();
		{
			std::osyncstream aout(std::cout);
			aout << "===================================================\n";
			aout << "================== NEW RELU LOOP ==================\n";
			aout << "LOOP INDEX: " << num << '\n';
			aout << "RELU RANDOM SEED: " << random_seed << std::endl;
			aout << "===================================================\n";
		}
		auto res = MNIST_Attack_ReLU(random_seed, mnist_train, mnist_valid, mnist_test, mnist_train_classes_w.GetWeights(), mnist_test_classes_w.GetWeights());
		std::ofstream log("exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__ReLU.log", std::ios::app);
		log << res.accuracy << '\t' << res.f1 << "\tA\t" << res.cwl2_success_rate << '\t' << res.bim_success_rate << "\tI\t" << res.cwl2_iters << '\t' << res.bim_iters << '\n';
		log.close();
	};

	auto ModelRunner_nmiReLUeq = [&](unsigned num) {
		locker.lock();
		unsigned random_seed = randevice();
		locker.unlock();
		{
			std::osyncstream aout(std::cout);
			aout << "===================================================\n";
			aout << "=============== NEW NMI-RELU-EQ LOOP ==============\n";
			aout << "LOOP INDEX: " << num << '\n';
			aout << "NMI-RELU-EQ RANDOM SEED: " << random_seed << std::endl;
			aout << "===================================================\n";
		}
		auto res = MNIST_Attack_nmiReLU_eq_nmi(random_seed, mnist_train, mnist_valid, mnist_test, mnist_train_classes_w.GetWeights(), mnist_test_classes_w.GetWeights());
		std::ofstream log("exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_EQ.log", std::ios::app);
		log << res.accuracy << '\t' << res.f1 << '\t' << res.stand_percent << '\t' << res.extra_percent << "\tA\t" << res.cwl2_success_rate << '\t' << res.bim_success_rate << "\tI\t" << res.cwl2_iters << '\t' << res.bim_iters << '\n';
		log.close();
	};

	auto ModelRunner_nmiReLUjn = [&](unsigned num) {
		locker.lock();
		unsigned random_seed = randevice();
		locker.unlock();
		{
			std::osyncstream aout(std::cout);
			aout << "===================================================\n";
			aout << "=============== NEW NMI-RELU-JN LOOP ==============\n";
			aout << "LOOP INDEX: " << num << '\n';
			aout << "NMI-RELU-JN RANDOM SEED: " << random_seed << std::endl;
			aout << "===================================================\n";
		}
		auto res = MNIST_Attack_nmiReLU_nmi(random_seed, mnist_train, mnist_valid, mnist_test, mnist_train_classes_w.GetWeights(), mnist_test_classes_w.GetWeights());
		std::ofstream log("exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_JN.log", std::ios::app);
		log << res.accuracy << '\t' << res.f1 << '\t' << res.stand_percent << '\t' << res.extra_percent << "\tA\t" << res.cwl2_success_rate << '\t' << res.bim_success_rate << "\tI\t" << res.cwl2_iters << '\t' << res.bim_iters << '\n';
		log.close();
	};

	auto ModelRunner_nm0ReLU = [&](unsigned num) {
		locker.lock();
		unsigned random_seed = randevice();
		locker.unlock();
		{
			std::osyncstream aout(std::cout);
			aout << "===================================================\n";
			aout << "================ NEW NM0-RELU LOOP ================\n";
			aout << "LOOP INDEX: " << num << '\n';
			aout << "NM0-RELU RANDOM SEED: " << random_seed << std::endl;
			aout << "===================================================\n";
		}
		auto res = MNIST_Attack_nmiReLU_nm0(random_seed, mnist_train, mnist_valid, mnist_test, mnist_train_classes_w.GetWeights(), mnist_test_classes_w.GetWeights());
		std::ofstream log("exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmReLU.log", std::ios::app);
		log << res.accuracy << '\t' << res.f1 << '\t' << res.stand_percent << '\t' << res.extra_percent << "\tA\t" << res.cwl2_success_rate << '\t' << res.bim_success_rate << "\tI\t" << res.cwl2_iters << '\t' << res.bim_iters << '\n';
		log.close();
	};

	auto Thread1 = [&]() {
		unsigned loop_id = 0;
		while (true) {
			ModelRunner_nm0ReLU(loop_id);
			ModelRunner_ReLU(loop_id);
			ModelRunner_nmiReLUeq(loop_id);
			ModelRunner_nmiReLUjn(loop_id);
			++loop_id;
		}
	};

	auto Thread2 = [&]() {
		unsigned loop_id = 0;
		while (true) {
			ModelRunner_ReLU(loop_id);
			ModelRunner_nm0ReLU(loop_id);
			ModelRunner_nmiReLUjn(loop_id);
			ModelRunner_nmiReLUeq(loop_id);
			++loop_id;
		}
	};

	std::thread wrk1(Thread1);
	std::thread wrk2(Thread2);

	wrk1.join();
	wrk2.join();
}


struct Attack_result { float cwl2_success_rate, bim_success_rate; unsigned cwl2_iters, bim_iters; };
static void CWL2_ReLU_StatePrinter(unsigned short binstep, unsigned iteration, float loss);
static void BIM_ReLU_StatePrinter(unsigned iteration, float loss);
static void CWL2_nmiReLUeq_StatePrinter(unsigned short binstep, unsigned iteration, float loss);
static void BIM_nmiReLUeq_StatePrinter(unsigned iteration, float loss);
static void CWL2_nmiReLUjn_StatePrinter(unsigned short binstep, unsigned iteration, float loss);
static void BIM_nmiReLUjn_StatePrinter(unsigned iteration, float loss);
static void CWL2_nm0ReLU_StatePrinter(unsigned short binstep, unsigned iteration, float loss);
static void BIM_nm0ReLU_StatePrinter(unsigned iteration, float loss);

static inline Attack_result Attack_ReLU(unsigned random_seed, const std::vector<float> &weights_store) {
	Attack_result result{0, 0, 0, 0};

	using nnReLU = nn::NNB_ReLUb<ATTACK_BATCH_SIZE, true>; // Using Kahan summation
	using nnInput = nn::NNB_Input_spyableB<ATTACK_BATCH_SIZE, true>; // Using Kahan summation
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(ATTACK_BATCH_SIZE);
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) { new(mem_ptr)nn::NNB_ConstInput; });
	nn::NeuronHoldingStaticLayer<nnInput> layer_inp(INPUT_SIZE, [&](nnInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nnInput([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity; for (size_t i = 0; i != capacity; ++i) storage[i] = &inputs_store[i][index];
		});
	});
	nn::NeuronHoldingStaticLayer<nnReLU> layer_hid1(RELU_LAYER_HIDDEN1_SIZE, [&](nnReLU *const mem_ptr, unsigned) { new(mem_ptr)nnReLU; });
	nn::NeuronHoldingStaticLayer<nnReLU> layer_hid2(LAYER_HIDDEN2_SIZE, [&](nnReLU *const mem_ptr, unsigned) { new(mem_ptr)nnReLU; });
	nn::NeuronHoldingStaticLayer<nnReLU> layer_out(10, [&](nnReLU *const mem_ptr, unsigned) { new(mem_ptr)nnReLU; });
	using NoOptim = nn::optimizers::GradientDescendent;
	NoOptim dummyOptimizer(0);
	using ConnecT = nn::NNB_Connection<NoOptim>;
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_bias_hid1(&layer_bias, &layer_hid1, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_bias_hid2(&layer_bias, &layer_hid2, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_bias_out(&layer_bias, &layer_out, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_inp_hid1(&layer_inp, &layer_hid1, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_hid1_hid2(&layer_hid1, &layer_hid2, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_hid2_out(&layer_hid2, &layer_out, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	// END MODEL DEFINE

	nn::LearnGuiderFwBPgThreadAble inferguider({ &layer_inp, &layer_hid1, &layer_hid2, &layer_out }, ATTACK_BATCH_SIZE, THREADS_COUNT); // Evaluator

	{ // Restoring weights
		using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>>::CBIterator;
		IterableAggregation<nn::interfaces::CBI *> weights_iter;
		weights_iter.AppendIterableItem<IterType>(connection_bias_hid1);
		weights_iter.AppendIterableItem<IterType>(connection_bias_hid2);
		weights_iter.AppendIterableItem<IterType>(connection_bias_out);
		weights_iter.AppendIterableItem<IterType>(connection_inp_hid1);
		weights_iter.AppendIterableItem<IterType>(connection_hid1_hid2);
		weights_iter.AppendIterableItem<IterType>(connection_hid2_out);

		auto cwght = weights_store.cbegin();
		for (auto ifc : weights_iter) {
			ifc->Weight(*cwght);
			++cwght;
		}
	}

	// CWL2 BLOCK
	{ 
		nn::reverse::CarliniWagnerL2Params cwl2params;
		cwl2params.box_min = 0.0f;
		cwl2params.box_max = 1.0f;
		cwl2params.loss1_scale_init = 1e-1f;
		cwl2params.binary_search_steps = 5;
		cwl2params.state_printing_period = LOG_INTERVAL;
		cwl2params.StatePrinter = &CWL2_ReLU_StatePrinter;
		//cwl2params.allow_early_stop = false;

		nn::reverse::CarliniWagnerL2ThreadAble cwl2(inferguider, cwl2params, ATTACK_BATCH_SIZE, THREADS_COUNT);

		std::mt19937 preudorandom(random_seed);
		std::uniform_real_distribution<float> randistributor(0.3f, 0.7f);
		std::uniform_int_distribution<unsigned short> randistributor_cls(0, 9);

		unsigned succedeed = 0;
		unsigned overall_iters = 0;

		for (unsigned short i = 0; i != ATTACK_REPEAT_COUNT; i++) {
			std::vector<std::vector<float>> sources(ATTACK_BATCH_SIZE, std::vector<float>(INPUT_SIZE));
			for (auto &bth : sources) {
				for (auto &pxl : bth) {
					pxl = randistributor(preudorandom);
				}
			}
			std::vector<unsigned short> targets(ATTACK_BATCH_SIZE);
			for (auto &tgt : targets) {
				tgt = randistributor_cls(preudorandom);
			}

			std::osyncstream(std::cout) << "ReLU:CWL2:: Begin...\n";

			unsigned iters;
			auto attack_results = cwl2.RunAttack(sources, targets, &iters);
			overall_iters += iters;

			for (auto &attk : attack_results) {
				if (attk.size()) {
					++succedeed;
				}
			}
		}

		result.cwl2_success_rate = (100.0f * succedeed) / (ATTACK_BATCH_SIZE * ATTACK_REPEAT_COUNT);
		result.cwl2_iters = overall_iters;

		std::osyncstream(std::cout) << "ReLU:CWL2:: SUCCESS RATE " << result.cwl2_success_rate << "%  Iters: " << result.cwl2_iters << '\n';
	}

	// BIM BLOCK
	{
		nn::reverse::BasicIterativeMethodParams bimparams;
		bimparams.box_min = 0.0f;
		bimparams.box_max = 1.0f;
		bimparams.max_iterations = 1500;
		bimparams.state_printing_period = LOG_INTERVAL;
		bimparams.StatePrinter = &BIM_ReLU_StatePrinter;
		//bimparams.allow_early_stop = false;

		nn::reverse::BasicIterativeMethodThreadAble bim(inferguider, bimparams, ATTACK_BATCH_SIZE, THREADS_COUNT);

		std::mt19937 preudorandom(random_seed);
		std::uniform_real_distribution<float> randistributor(0.3f, 0.7f);
		std::uniform_int_distribution<unsigned short> randistributor_cls(0, 9);

		unsigned succedeed = 0;
		unsigned overall_iters = 0;

		for (unsigned short i = 0; i != ATTACK_REPEAT_COUNT; i++) {
			std::vector<std::vector<float>> sources(ATTACK_BATCH_SIZE, std::vector<float>(INPUT_SIZE));
			for (auto &bth : sources) {
				for (auto &pxl : bth) {
					pxl = randistributor(preudorandom);
				}
			}
			std::vector<unsigned short> targets(ATTACK_BATCH_SIZE);
			for (auto &tgt : targets) {
				tgt = randistributor_cls(preudorandom);
			}

			std::osyncstream(std::cout) << "ReLU:BIM:: Begin...\n";

			unsigned iters;
			auto attack_results = bim.RunAttack(sources, targets, &iters);
			overall_iters += iters;

			for (auto succ : attack_results) {
				if (succ) {
					++succedeed;
				}
			}
		}

		result.bim_success_rate = (100.0f * succedeed) / (ATTACK_BATCH_SIZE * ATTACK_REPEAT_COUNT);
		result.bim_iters = overall_iters;

		std::osyncstream(std::cout) << "ReLU:BIM:: SUCCESS RATE " << result.bim_success_rate << "%  Iters: " << result.bim_iters << '\n';
	}

	return result;
}

static inline Attack_result Attack_nmiReLU(unsigned random_seed, bool nmi, bool with_equivalent, const std::vector<float> &weights_store) {
	Attack_result result{ 0, 0, 0, 0 };

	using nnNmReLU = nn::NNB_nm1iReLUb<BATCH_SIZE, true, true>; // Non-monotonic Immortal ReLU, using Kahan summation
	using nnInput = nn::NNB_Input_spyableB<ATTACK_BATCH_SIZE, true>; // Using Kahan summation
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(ATTACK_BATCH_SIZE);
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) { new(mem_ptr)nn::NNB_ConstInput; });
	nn::NeuronHoldingStaticLayer<nnInput> layer_inp(INPUT_SIZE, [&](nnInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nnInput([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity; for (size_t i = 0; i != capacity; ++i) storage[i] = &inputs_store[i][index];
		});
	});
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer1_nm_hid(NM_LAYER_HIDDEN1_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) { new(mem_ptr)nnNmReLU(NM1_MAX_VALUE); });
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer2_nm_hid(LAYER_HIDDEN2_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) { new(mem_ptr)nnNmReLU(NM1_MAX_VALUE); });
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer3_nm_out(10, [&](nnNmReLU *const mem_ptr, unsigned) { new(mem_ptr)nnNmReLU(NM1_MAX_VALUE); });
	using NoOptim = nn::optimizers::GradientDescendent;
	NoOptim dummyOptimizer(0);
	using ConnecT = nn::NNB_Connection<NoOptim>;
	nn::DenseLayerStaticConnectomHolder<ConnecT> connections_bias_nmhid1(&layer_bias, &layer1_nm_hid, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<NoOptim>(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connections_bias_nmhid2(&layer_bias, &layer2_nm_hid, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connections_bias_nmout(&layer_bias, &layer3_nm_out, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_inp_nmhid1(&layer_inp, &layer1_nm_hid, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_nmhid1_nmhid2(&layer1_nm_hid, &layer2_nm_hid, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	nn::DenseLayerStaticConnectomHolder<ConnecT> connection_nmhid2_nmout(&layer2_nm_hid, &layer3_nm_out, [&](ConnecT *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)ConnecT(from, to, &dummyOptimizer);
	});
	// END MODEL DEFINE

	nn::LearnGuiderFwBPgThreadAble inferguider({ &layer_inp, &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }, ATTACK_BATCH_SIZE, THREADS_COUNT); // Evaluator

	{ // Restoring weights
		using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>>::CBIterator;
		IterableAggregation<nn::interfaces::CBI *> weights_iter;
		weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid1);
		weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid2);
		weights_iter.AppendIterableItem<IterType>(connections_bias_nmout);
		weights_iter.AppendIterableItem<IterType>(connection_inp_nmhid1);
		weights_iter.AppendIterableItem<IterType>(connection_nmhid1_nmhid2);
		weights_iter.AppendIterableItem<IterType>(connection_nmhid2_nmout);

		auto cwght = weights_store.cbegin();
		for (auto ifc : weights_iter) {
			ifc->Weight(*cwght);
			++cwght;
		}
	}

	// CWL2 BLOCK
	{
		nn::reverse::CarliniWagnerL2Params cwl2params;
		cwl2params.box_min = 0.0f;
		cwl2params.box_max = 1.0f;
		cwl2params.loss1_scale_init = 1e-1f;
		cwl2params.binary_search_steps = 5;
		cwl2params.state_printing_period = LOG_INTERVAL;
		cwl2params.StatePrinter = (nmi ? (with_equivalent ? &CWL2_nmiReLUeq_StatePrinter : &CWL2_nmiReLUjn_StatePrinter) : &CWL2_nm0ReLU_StatePrinter);
		//cwl2params.allow_early_stop = false;

		nn::reverse::CarliniWagnerL2ThreadAble cwl2(inferguider, cwl2params, ATTACK_BATCH_SIZE, THREADS_COUNT);

		std::mt19937 preudorandom(random_seed);
		std::uniform_real_distribution<float> randistributor(0.3f, 0.7f);
		std::uniform_int_distribution<unsigned short> randistributor_cls(0, 9);

		unsigned succedeed = 0;
		unsigned overall_iters = 0;

		for (unsigned short i = 0; i != ATTACK_REPEAT_COUNT; i++) {
			std::vector<std::vector<float>> sources(ATTACK_BATCH_SIZE, std::vector<float>(INPUT_SIZE));
			for (auto &bth : sources) {
				for (auto &pxl : bth) {
					pxl = randistributor(preudorandom);
				}
			}
			std::vector<unsigned short> targets(ATTACK_BATCH_SIZE);
			for (auto &tgt : targets) {
				tgt = randistributor_cls(preudorandom);
			}

			std::osyncstream(std::cout) << (nmi ? (with_equivalent ? "nmiReLU-EQ" : "nmiReLU-JN") : "nm0ReLU") << ":CWL2:: Begin...\n";

			unsigned iters;
			auto attack_results = cwl2.RunAttack(sources, targets, &iters);
			overall_iters += iters;

			for (auto &attk : attack_results) {
				if (attk.size()) {
					++succedeed;
				}
			}
		}

		result.cwl2_success_rate = (100.0f * succedeed) / (ATTACK_BATCH_SIZE * ATTACK_REPEAT_COUNT);
		result.cwl2_iters = overall_iters;

		std::osyncstream(std::cout) << (nmi ? (with_equivalent ? "nmiReLU-EQ" : "nmiReLU-JN") : "nm0ReLU") << ":CWL2:: SUCCESS RATE " << result.cwl2_success_rate << "%  Iters: " << result.cwl2_iters << '\n';
	}

	// BIM BLOCK
	{
		nn::reverse::BasicIterativeMethodParams bimparams;
		bimparams.box_min = 0.0f;
		bimparams.box_max = 1.0f;
		bimparams.state_printing_period = LOG_INTERVAL;
		bimparams.StatePrinter = (nmi ? (with_equivalent ? &BIM_nmiReLUeq_StatePrinter : &BIM_nmiReLUjn_StatePrinter) : &BIM_nm0ReLU_StatePrinter);
		//bimparams.allow_early_stop = false;

		nn::reverse::BasicIterativeMethodThreadAble bim(inferguider, bimparams, ATTACK_BATCH_SIZE, THREADS_COUNT);

		std::mt19937 preudorandom(random_seed);
		std::uniform_real_distribution<float> randistributor(0.3f, 0.7f);
		std::uniform_int_distribution<unsigned short> randistributor_cls(0, 9);

		unsigned succedeed = 0;
		unsigned overall_iters = 0;

		for (unsigned short i = 0; i != ATTACK_REPEAT_COUNT; i++) {
			std::vector<std::vector<float>> sources(ATTACK_BATCH_SIZE, std::vector<float>(INPUT_SIZE));
			for (auto &bth : sources) {
				for (auto &pxl : bth) {
					pxl = randistributor(preudorandom);
				}
			}
			std::vector<unsigned short> targets(ATTACK_BATCH_SIZE);
			for (auto &tgt : targets) {
				tgt = randistributor_cls(preudorandom);
			}

			std::osyncstream(std::cout) << (nmi ? (with_equivalent ? "nmiReLU-EQ" : "nmiReLU-JN") : "nm0ReLU") << ":BIM:: Begin...\n";

			unsigned iters;
			auto attack_results = bim.RunAttack(sources, targets, &iters);
			overall_iters += iters;

			for (auto succ : attack_results) {
				if (succ) {
					++succedeed;
				}
			}
		}

		result.bim_success_rate = (100.0f * succedeed) / (ATTACK_BATCH_SIZE * ATTACK_REPEAT_COUNT);
		result.bim_iters = overall_iters;

		std::osyncstream(std::cout) << (nmi ? (with_equivalent ? "nmiReLU-EQ" : "nmiReLU-JN") : "nm0ReLU") << ":BIM:: SUCCESS RATE " << result.bim_success_rate << "%  Iters: " << result.bim_iters << '\n';
	}

	return result;
}

static inline MNIST_Attack_ReLU_result MNIST_Attack_ReLU(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train_sample, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test) {
	std::mt19937 preudorandom(random_seed);
	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);

	using nnReLU = nn::NNB_ReLUb<BATCH_SIZE, true>; // Using Kahan summation

	std::vector<MNIST_entry> mnist_train = mnist_train_sample;

	// Input image (like vector, flatten)
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(BATCH_SIZE);

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(INPUT_SIZE, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnReLU> layer_hid1(RELU_LAYER_HIDDEN1_SIZE, [&](nnReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnReLU;
	});

	// Hidden layer 2
	nn::NeuronHoldingStaticLayer<nnReLU> layer_hid2(LAYER_HIDDEN2_SIZE, [&](nnReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnReLU;
	});

	// Output layer
	nn::NeuronHoldingStaticLayer<nnReLU> layer_out(10, [&](nnReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnReLU;
	});

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);

	// Connections
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_bias_hid1(&layer_bias, &layer_hid1, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_bias_hid2(&layer_bias, &layer_hid2, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_bias_out(&layer_bias, &layer_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// Input to
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_hid1(&layer_inp, &layer_hid1, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_hid1_hid2(&layer_hid1, &layer_hid2, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_hid2_out(&layer_hid2, &layer_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// Combining connectoms into one structure
	using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>>::CBIterator;
	IterableAggregation<nn::interfaces::CBI *> weights_iter;
	weights_iter.AppendIterableItem<IterType>(connection_bias_hid1);
	weights_iter.AppendIterableItem<IterType>(connection_bias_hid2);
	weights_iter.AppendIterableItem<IterType>(connection_bias_out);
	weights_iter.AppendIterableItem<IterType>(connection_inp_hid1);
	weights_iter.AppendIterableItem<IterType>(connection_hid1_hid2);
	weights_iter.AppendIterableItem<IterType>(connection_hid2_out);

	nn::LearnGuiderFwBPgThreadAble learnguider_multhr({ &layer_hid1, &layer_hid2, &layer_out }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training
	nn::LearnGuiderFwBPg learnguider_single({ &layer_hid1, &layer_hid2, &layer_out }, BATCH_SIZE); // Evaluator for validating & testing

	const float UINT8_NORM = 1.0f / 255.0f;

	// Learning
	std::vector<std::vector<float>> perfect_out_store(BATCH_SIZE, std::vector<float>(10));
	std::vector<unsigned short> perfect_outs(BATCH_SIZE);

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_train_calculator(weights_train);
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator(weights_test);

	std::atomic_flag data_is_ready_for_workers;
	std::atomic_uint workers_is_ready_for_data;
	bool threads_run = true;

	bool calc_loss = false;
	std::atomic<float> threads_loss;

	auto TrainWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_multhr.GetRequiredCachesCount());
		nn::errcalc::ErrorCalcSoftMAX softmax_calculator(10);

		float loss;
		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			loss = 0.0f;
			// Learning
			learnguider_multhr.WorkerDoForward(worker_id, &caches[0]);
			loss += learnguider_multhr.WorkerFillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
			learnguider_multhr.WorkerDoBackward(worker_id, &caches[0]);

			// Loss & local accuracy
			if (calc_loss) {
				f1score_train_calculator.AppendResultsThreadSafe(learnguider_multhr.GetOutputs(), perfect_out_store, worker_id, THREADS_COUNT);
				threads_loss.fetch_add(loss, std::memory_order_relaxed);
			}

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	std::vector<std::thread> workers;
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TrainWorkerThread, i);
	}

	std::osyncstream(std::cout) << "ReLU: Training... " << '\n';

	unsigned current_threads_count;
	unsigned log_iterations = 0;
	unsigned batch_iterations = 0;
	unsigned batch_iterations_all = mnist_train.size() / BATCH_SIZE;
	for (unsigned epoch = 0; epoch != EPOCHS; ++epoch) {
		std::shuffle(mnist_train.begin(), mnist_train.end(), preudorandom);

		auto datase_current_pos = mnist_train.cbegin();
		auto datase_end = mnist_train.cend();

		batch_iterations = 0;
		// Training epoch
		do {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				auto &out = perfect_out_store[batch_i];
				std::fill(out.begin(), out.end(), 0.0f);
				out[datase_current_pos->label] = 1.0f;
			}
			++batch_iterations;

			calc_loss = false;
			if (++log_iterations == LOG_INTERVAL) {
				log_iterations = 0;
				calc_loss = true;
			}

			data_is_ready_for_workers.test_and_set(std::memory_order_release);
			data_is_ready_for_workers.notify_all();

			// Training is performing now

			current_threads_count = 0;
			do {
				workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
				current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
			} while (current_threads_count != THREADS_COUNT);
			workers_is_ready_for_data.store(0, std::memory_order_release);

			if (calc_loss) {
				float loss = threads_loss.load(std::memory_order_relaxed);
				threads_loss.store(0.0f, std::memory_order_relaxed);
				float f1 = f1score_train_calculator.CalcF1();
				float accuracy = f1score_train_calculator.CalcAccuracy();
				f1score_train_calculator.Reset();
				loss /= BATCH_SIZE;
				std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "ReLU: Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << accuracy << "%  F1(local): " << f1 << "\n";
			}
		} while (datase_current_pos != datase_end);

		// Fast validation after epoch
		datase_current_pos = mnist_valid.cbegin();
		datase_end = mnist_valid.cend();
		while (true) {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				perfect_outs[batch_i] = datase_current_pos->label;
			}
			if (datase_current_pos == datase_end) {
				break;
			}

			learnguider_single.DoForward();

			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
				f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_single.GetOutputs(), batch_i), perfect_outs[batch_i]);
			}
		}
		std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "ReLU: Accuracy (validation dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
		f1score_test_calculator.Reset();
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();
	// Ensuring all released
	for (auto &thr : workers) {
		thr.join();
	}


	std::osyncstream(std::cout) << "ReLU: Accuracy calculating...\n";

	// Accuracy calc
	auto TestWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_multhr.GetRequiredCachesCount(false));
		std::vector<float> *caches_ptr = caches.size() ? &caches[0] : nullptr;

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			learnguider_multhr.WorkerDoForward(worker_id, caches_ptr);
			f1score_test_calculator.AppendResultsThreadSafe(learnguider_multhr.GetOutputs(), perfect_outs, worker_id, THREADS_COUNT);

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	// Threads prepare
	data_is_ready_for_workers.clear();
	threads_run = true;

	workers.clear();
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TestWorkerThread, i);
	}

	unsigned batch_count = 0;
	auto test_current_pos = mnist_test.cbegin();
	auto test_end = mnist_test.cend();
	batch_iterations_all = mnist_test.size() / BATCH_SIZE;

	log_iterations = 0;
	while (true) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++test_current_pos) {
			if (test_current_pos == test_end) {
				break;
			}
			auto iter = inputs_store[batch_i].begin();
			for (auto item : test_current_pos->input) {
				*iter = UINT8_NORM * item;
				++iter;
			}
			perfect_outs[batch_i] = test_current_pos->label;
		}
		if (test_current_pos == test_end) {
			break;
		}
		++batch_count;
		if (++log_iterations == LOG_INTERVAL) {
			log_iterations = 0;
			std::osyncstream(std::cout) << "ReLU: Batch: " << batch_count << '/' << batch_iterations_all << '\n';
		}

		data_is_ready_for_workers.test_and_set(std::memory_order_release);
		data_is_ready_for_workers.notify_all();

		// Evaluating is performing now

		current_threads_count = 0;
		do {
			workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
			current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
		} while (current_threads_count != THREADS_COUNT);
		workers_is_ready_for_data.store(0, std::memory_order_release);
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();

	float accuracy = f1score_test_calculator.CalcAccuracy();
	float f1 = f1score_test_calculator.CalcF1();
	std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "ReLU: Accuracy (test dataset): " << accuracy << "%  F1: " << f1 << "\n";

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}

	std::vector<float> weights_store;
	for (auto ifc : weights_iter) {
		weights_store.push_back(ifc->Weight());
	}

	std::uniform_int_distribution<unsigned> randistributor_seed(0, std::numeric_limits<unsigned>::max());

	auto attack_success_rate = Attack_ReLU(randistributor_seed(preudorandom), weights_store);

	return MNIST_Attack_ReLU_result(accuracy, f1, attack_success_rate.cwl2_success_rate, attack_success_rate.bim_success_rate, attack_success_rate.cwl2_iters, attack_success_rate.bim_iters);
}

static inline MNIST_Attack_nmiReLU_result MNIST_Attack_nmiReLU_eq_nmi(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train_sample, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test) {
	std::mt19937 preudorandom(random_seed);

	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);
	std::uniform_real_distribution<float> randistributor_diviation(-0.0025f, 0.0025f);

	std::vector<MNIST_entry> mnist_train = mnist_train_sample;

	using nnEqReLU = nn::NNB_ilReLUb<BATCH_SIZE, true>; // Immortal Limited ReLU, using Kahan summation
	using nnNmReLU = nn::NNB_nm1iReLUb<BATCH_SIZE, true, true>; // Non-monotonic Immortal ReLU, using Kahan summation
	using nnLinSumm = nn::NNB_LinearSlimT<BATCH_SIZE, true>; // Linear summator, using Kahan

	// Input image (like vector, flatten)
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(BATCH_SIZE);

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(INPUT_SIZE, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Non-monotonic Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer1_nm_hid(NM_LAYER_HIDDEN1_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Hidden layer 2
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer2_nm_hid(LAYER_HIDDEN2_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Output layer
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer3_nm_out(10, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});


	// Equivalent Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer1_eq_hid_inc(NM_LAYER_HIDDEN1_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer1_eq_hid_dec(NM_LAYER_HIDDEN1_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NNB_LayersAggregator layer1_eq_hid({ &layer1_eq_hid_inc, &layer1_eq_hid_dec });
	// Heading summator
	nn::NeuronHoldingStaticLayer<nnLinSumm> layer15_eq_hid_head(NM_LAYER_HIDDEN1_SIZE, [&](nnLinSumm *const mem_ptr, unsigned) {
		new(mem_ptr)nnLinSumm;
	});

	// Equivalent Hidden layer 2
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer2_eq_hid_inc(LAYER_HIDDEN2_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer2_eq_hid_dec(LAYER_HIDDEN2_SIZE, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NNB_LayersAggregator layer2_eq_hid({ &layer2_eq_hid_inc, &layer2_eq_hid_dec });
	// Heading summator
	nn::NeuronHoldingStaticLayer<nnLinSumm> layer25_eq_hid_head(LAYER_HIDDEN2_SIZE, [&](nnLinSumm *const mem_ptr, unsigned) {
		new(mem_ptr)nnLinSumm;
	});


	// Equivalent Output layer
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer3_eq_out_inc(10, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NeuronHoldingStaticLayer<nnEqReLU> layer3_eq_out_dec(10, [&](nnEqReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnEqReLU(NM1_MAX_VALUE);
	});
	nn::NNB_LayersAggregator layer3_eq_out({ &layer3_eq_out_inc, &layer3_eq_out_dec });
	// Heading summator
	nn::NeuronHoldingStaticLayer<nnLinSumm> layer35_eq_out_head(10, [&](nnLinSumm *const mem_ptr, unsigned) {
		new(mem_ptr)nnLinSumm;
	});

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);

	// Connections of Non-Monotinic (weights initialization is not required if equivalent trained first)
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid1(&layer_bias, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid2(&layer_bias, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmout(&layer_bias, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	// Non-Monotinic FeedForward connections
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_nmhid1(&layer_inp, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid1_nmhid2(&layer1_nm_hid, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid2_nmout(&layer2_nm_hid, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// Combining connectoms into one structure
	using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>>::CBIterator;
	IterableAggregation<nn::interfaces::CBI *> weights_iter;
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid1);
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid2);
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmout);
	weights_iter.AppendIterableItem<IterType>(connection_inp_nmhid1);
	weights_iter.AppendIterableItem<IterType>(connection_nmhid1_nmhid2);
	weights_iter.AppendIterableItem<IterType>(connection_nmhid2_nmout);

	// Connections of Equivalent
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_eqhid1(&layer_bias, &layer1_eq_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_eqhid2(&layer_bias, &layer2_eq_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_eqout(&layer_bias, &layer3_eq_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor_diviation(preudorandom));
	});
	// Equivalent FeedForward connections
	// Input -> Hidden-1 EQ1
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_eqhid1_inc(&layer_inp, &layer1_eq_hid_inc, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	// Input -> Hidden-1 EQ2
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_eqhid1_dec(&layer_inp, &layer1_eq_hid_dec, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -randistributor(preudorandom));
	});
	// (Hidden-1 EQ1 + Hidden-1 EQ2) -> Hidden-1 Head
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connections_inp_eqhid1_head(&layer1_eq_hid, &layer15_eq_hid_head, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});
	// Hidden-1 Head -> Hidden-2 EQ1
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid1_eqhid2_inc(&layer15_eq_hid_head, &layer2_eq_hid_inc, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	// Hidden-1 Head -> Hidden-2 EQ2
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid1_eqhid2_dec(&layer15_eq_hid_head, &layer2_eq_hid_dec, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -randistributor(preudorandom));
	});
	// (Hidden-2 EQ1 + Hidden-2 EQ2) -> Hidden-2 Head
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connections_eqhid1_eqhid2_head(&layer2_eq_hid, &layer25_eq_hid_head, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});

	// Hidden-2 Head -> Out EQ1
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid2_out_inc(&layer25_eq_hid_head, &layer3_eq_out_inc, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	// Hidden-2 Head -> Out EQ2
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_eqhid2_out_dec(&layer25_eq_hid_head, &layer3_eq_out_dec, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -randistributor(preudorandom));
	});
	// (Out EQ1 + Out EQ2) -> Out Head
	nn::SparceLayerStaticConnectomHolder2Mult<nn::NNB_StraightConnection> connection_eqhid2_out_head(&layer3_eq_out, &layer35_eq_out_head, [&](nn::NNB_StraightConnection *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_StraightConnection(from, to);
	});

	// Projecting accessory
	nn::Monotonic2FieldsHeuristicsEqExV2 mon2filds(PROJECTION_THRESHOLD);
	nn::Monotonic2FieldsProjectingAccessory projector_lr1(&layer1_nm_hid, &layer1_eq_hid, &mon2filds);
	nn::Monotonic2FieldsProjectingAccessory projector_lr2(&layer2_nm_hid, &layer2_eq_hid, &mon2filds);
	nn::Monotonic2FieldsProjectingAccessory projector_lr3(&layer3_nm_out, &layer3_eq_out, &mon2filds);


	nn::LearnGuiderFwBPgThreadAble learnguider_mtr_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training
	nn::LearnGuiderFwBPgThreadAble learnguider_mtr_equiv({ &layer1_eq_hid, &layer15_eq_hid_head, &layer2_eq_hid, &layer25_eq_hid_head, &layer3_eq_out, &layer35_eq_out_head }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training

	nn::LearnGuiderFwBPg learnguider_sing_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }); // Evaluator for validating & testing
	nn::LearnGuiderFwBPg learnguider_sing_equiv({ &layer1_eq_hid, &layer15_eq_hid_head, &layer2_eq_hid, &layer25_eq_hid_head, &layer3_eq_out, &layer35_eq_out_head }); // Evaluator for validating & testing

	const float UINT8_NORM = 1.0f / 255.0f;

	// Learning
	std::vector<std::vector<float>> perfect_out_store(BATCH_SIZE, std::vector<float>(10));
	std::vector<unsigned short> perfect_outs(BATCH_SIZE);

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_train_calculator(weights_train);
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator(weights_test);

	std::osyncstream(std::cout) << "nmiReLU-EQ: Training... " << '\n';

	std::atomic_flag data_is_ready_for_workers;
	std::atomic_uint workers_is_ready_for_data;
	bool threads_run = true;

	bool learn_nonmonotonic = false;
	bool calc_loss = false;
	std::atomic<float> threads_loss;

	auto TrainWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount());
		nn::errcalc::ErrorCalcSoftMAX softmax_calculator(10);

		float loss;
		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			loss = 0.0f;
			// Learning
			if (learn_nonmonotonic) {
				learnguider_mtr_nm.WorkerDoForward(worker_id, &caches[0]);
				loss += learnguider_mtr_nm.WorkerFillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
				learnguider_mtr_nm.WorkerDoBackward(worker_id, &caches[0]);
			} else {
				learnguider_mtr_equiv.WorkerDoForward(worker_id, &caches[0]);
				loss += learnguider_mtr_equiv.WorkerFillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
				learnguider_mtr_equiv.WorkerDoBackward(worker_id, &caches[0]);
			}

			// Loss & local accuracy
			if (calc_loss) {
				f1score_train_calculator.AppendResultsThreadSafe((learn_nonmonotonic ? learnguider_mtr_nm.GetOutputs() : learnguider_mtr_equiv.GetOutputs()), perfect_out_store, worker_id, THREADS_COUNT);
				threads_loss.fetch_add(loss, std::memory_order_relaxed);
			}

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	std::vector<std::thread> workers;
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TrainWorkerThread, i);
	}

	unsigned current_threads_count;
	unsigned log_iterations = 0;
	unsigned batch_iterations = 0;
	unsigned batch_iterations_all = mnist_train.size() / BATCH_SIZE;
	for (unsigned epoch = 0; epoch != EPOCHS; ++epoch) {
		std::shuffle(mnist_train.begin(), mnist_train.end(), preudorandom);

		auto datase_current_pos = mnist_train.cbegin();
		auto datase_end = mnist_train.cend();

		batch_iterations = 0;
		// Training epoch
		do {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				auto &out = perfect_out_store[batch_i];
				std::fill(out.begin(), out.end(), 0.0f);
				out[datase_current_pos->label] = 1.0f;
			}
			++batch_iterations;

			calc_loss = false;
			if (++log_iterations == LOG_INTERVAL) {
				log_iterations = 0;
				calc_loss = true;
			}

			data_is_ready_for_workers.test_and_set(std::memory_order_release);
			data_is_ready_for_workers.notify_all();

			// Training is performing now

			current_threads_count = 0;
			do {
				workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
				current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
			} while (current_threads_count != THREADS_COUNT);
			workers_is_ready_for_data.store(0, std::memory_order_release);

			if (calc_loss) {
				float loss = threads_loss.load(std::memory_order_relaxed);
				threads_loss.store(0.0f, std::memory_order_relaxed);
				loss /= BATCH_SIZE;
				std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nmiReLU-EQ: Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << f1score_train_calculator.CalcAccuracy() << "%  F1(local): " << f1score_train_calculator.CalcF1() << "\n";
				f1score_train_calculator.Reset();
			}
		} while (datase_current_pos != datase_end);

		// Fast validation after epoch
		datase_current_pos = mnist_valid.cbegin();
		datase_end = mnist_valid.cend();
		while (true) {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				perfect_outs[batch_i] = datase_current_pos->label;
			}
			if (datase_current_pos == datase_end) {
				break;
			}

			if (learn_nonmonotonic) {
				learnguider_sing_nm.DoForward();
				for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
					f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_sing_nm.GetOutputs(), batch_i), perfect_outs[batch_i]);
				}
			} else {
				learnguider_sing_equiv.DoForward();
				for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
					f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_sing_equiv.GetOutputs(), batch_i), perfect_outs[batch_i]);
				}
			}
		}
		std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nmiReLU-EQ: " << (learn_nonmonotonic ? "<NON-MONOTONIC>" : "<EQUIVALENT>") << "\t\t\tAccuracy (validation dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
		f1score_test_calculator.Reset();

		if (epoch == 1) {
			std::osyncstream(std::cout) << "nmiReLU-EQ: Performing EQUIVALENT => NON-MONOTONIC projecting...\n";
			projector_lr1.Perform2to1LossyCompression();
			projector_lr2.Perform2to1LossyCompression();
			projector_lr3.Perform2to1LossyCompression();
			learn_nonmonotonic = true;
		}
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();
	// Ensuring all released
	for (auto &thr : workers) {
		thr.join();
	}

	// Non-monotonity usage stats: minus_count, plus_count, min, summ, max
	std::map<const nnNmReLU *, std::tuple<unsigned, unsigned, unsigned, unsigned>> fu_both_neurons;

	auto NonMonotonityStatProc = [&](nnNmReLU *nrn) {
		auto iter = fu_both_neurons.find(nrn);
		if (iter != fu_both_neurons.end()) {
			float accval;
			for (unsigned batch_i = 0; batch_i != nrn->GetCurrentBatchSize(); ++batch_i) {
				accval = nrn->RealAccumulatorValue(batch_i);
				if (accval < 0) {
					++std::get<0>(iter->second);
					if (accval < -NM1_MAX_VALUE) {
						++std::get<1>(iter->second);
					}
				} else {
					++std::get<2>(iter->second);
					if (nrn->RealAccumulatorValue(batch_i) > NM1_MAX_VALUE) {
						++std::get<3>(iter->second);
					}
				}
			}
		} else {
			float accval;
			{
				accval = nrn->RealAccumulatorValue();
				fu_both_neurons.emplace(nrn, std::make_tuple((unsigned)(accval < 0), (unsigned)(accval < -NM1_MAX_VALUE), (unsigned)(accval >= 0), (unsigned)(accval > NM1_MAX_VALUE)));
			}
			iter = fu_both_neurons.find(nrn);
			for (unsigned batch_i = 1; batch_i < nrn->GetCurrentBatchSize(); ++batch_i) {
				accval = nrn->RealAccumulatorValue(batch_i);
				if (accval < 0) {
					++std::get<0>(iter->second);
					if (accval < -NM1_MAX_VALUE) {
						++std::get<1>(iter->second);
					}
				} else {
					++std::get<2>(iter->second);
					if (nrn->RealAccumulatorValue(batch_i) > NM1_MAX_VALUE) {
						++std::get<3>(iter->second);
					}
				}
			}
		}
	};


	std::osyncstream(std::cout) << "nmiReLU-EQ: Accuracy calculating...\n";

	// Accuracy calc
	auto TestWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount(false));
		std::vector<float> *caches_ptr = caches.size() ? &caches[0] : nullptr;

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			learnguider_mtr_nm.WorkerDoForward(worker_id, caches_ptr);
			f1score_test_calculator.AppendResultsThreadSafe(learnguider_mtr_nm.GetOutputs(), perfect_outs, worker_id, THREADS_COUNT);

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	// Threads prepare
	data_is_ready_for_workers.clear();
	threads_run = true;

	workers.clear();
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TestWorkerThread, i);
	}

	// Accuracy calc
	unsigned batch_count = 0;
	auto test_current_pos = mnist_test.cbegin();
	auto test_end = mnist_test.cend();
	batch_iterations_all = mnist_test.size() / BATCH_SIZE;

	log_iterations = 0;
	while (true) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++test_current_pos) {
			if (test_current_pos == test_end) {
				break;
			}
			auto iter = inputs_store[batch_i].begin();
			for (auto item : test_current_pos->input) {
				*iter = UINT8_NORM * item;
				++iter;
			}
			perfect_outs[batch_i] = test_current_pos->label;
		}
		if (test_current_pos == test_end) {
			break;
		}
		++batch_count;
		if (++log_iterations == LOG_INTERVAL) {
			log_iterations = 0;
			std::osyncstream(std::cout) << "nmiReLU-EQ: Batch: " << batch_count << '/' << batch_iterations_all << '\n';
		}

		data_is_ready_for_workers.test_and_set(std::memory_order_release);
		data_is_ready_for_workers.notify_all();

		// Evaluating is performing now

		current_threads_count = 0;
		do {
			workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
			current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
		} while (current_threads_count != THREADS_COUNT);
		workers_is_ready_for_data.store(0, std::memory_order_release);

		// Grab non-monotonity stat
		for (auto lr : learnguider_mtr_nm.GetLayers()) {
			for (auto nrn : lr->Neurons()) {
				NonMonotonityStatProc(dynamic_cast<nnNmReLU *>(nrn));
			}
		}
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();

	float stand_percent = 0.0f;
	float extra_percent = 0.0f;
	for (auto &iter : fu_both_neurons) {
		unsigned smin = std::min(std::get<0>(iter.second), std::get<2>(iter.second));
		unsigned smax = std::max(std::get<0>(iter.second), std::get<2>(iter.second));
		unsigned emin = std::min(std::get<1>(iter.second), std::get<3>(iter.second));
		unsigned emax = std::max(std::get<1>(iter.second), std::get<3>(iter.second));

		stand_percent += static_cast<float>(smin) / smax;
		extra_percent += static_cast<float>(emin) / emax;

	}
	stand_percent *= 100.0f / fu_both_neurons.size();
	extra_percent *= 100.0f / fu_both_neurons.size();

	float accuracy = f1score_test_calculator.CalcAccuracy();
	float f1 = f1score_test_calculator.CalcF1();
	std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nmiReLU-EQ: Accuracy (test dataset): " << accuracy << "%  F1: " << f1 << "  Both-f-usg:: stand: " << stand_percent << "%  extra: " << extra_percent << "%\n";

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}

	std::vector<float> weights_store;
	for (auto ifc : weights_iter) {
		weights_store.push_back(ifc->Weight());
	}

	std::uniform_int_distribution<unsigned> randistributor_seed(0, std::numeric_limits<unsigned>::max());

	auto attack_success_rate = Attack_nmiReLU(randistributor_seed(preudorandom), true, true, weights_store);

	return MNIST_Attack_nmiReLU_result{ accuracy, f1, stand_percent, extra_percent, attack_success_rate.cwl2_success_rate, attack_success_rate.bim_success_rate, attack_success_rate.cwl2_iters, attack_success_rate.bim_iters };
}

static inline MNIST_Attack_nmiReLU_result MNIST_Attack_nmiReLU_nmi(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train_sample, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test) {
	std::mt19937 preudorandom(random_seed);

	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);
	std::uniform_real_distribution<float> randistributor_diviation(-0.0025f, 0.0025f);

	std::vector<MNIST_entry> mnist_train = mnist_train_sample;

	using nnNmReLU = nn::NNB_nm1iReLUb<BATCH_SIZE, true, true>; // Non-monotonic Immortal ReLU, using Kahan summation
	using nnLinSumm = nn::NNB_LinearSlimT<BATCH_SIZE, true>; // Linear summator, using Kahan

	// Input image (like vector, flatten)
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(BATCH_SIZE);

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(INPUT_SIZE, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Non-monotonic Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer1_nm_hid(NM_LAYER_HIDDEN1_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Hidden layer 2
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer2_nm_hid(LAYER_HIDDEN2_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Output layer
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer3_nm_out(10, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);

	// Connections of Non-Monotinic (weights initialization is not required if equivalent trained first)
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid1(&layer_bias, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid2(&layer_bias, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmout(&layer_bias, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	// Non-Monotinic FeedForward connections
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_nmhid1(&layer_inp, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid1_nmhid2(&layer1_nm_hid, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid2_nmout(&layer2_nm_hid, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// Combining connectoms into one structure
	using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>>::CBIterator;
	IterableAggregation<nn::interfaces::CBI *> weights_iter;
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid1);
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid2);
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmout);
	weights_iter.AppendIterableItem<IterType>(connection_inp_nmhid1);
	weights_iter.AppendIterableItem<IterType>(connection_nmhid1_nmhid2);
	weights_iter.AppendIterableItem<IterType>(connection_nmhid2_nmout);


	nn::LearnGuiderFwBPgThreadAble learnguider_mtr_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training

	nn::LearnGuiderFwBPg learnguider_sing_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }); // Evaluator for validating & testing

	const float UINT8_NORM = 1.0f / 255.0f;

	// Learning
	std::vector<std::vector<float>> perfect_out_store(BATCH_SIZE, std::vector<float>(10));
	std::vector<unsigned short> perfect_outs(BATCH_SIZE);

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_train_calculator(weights_train);
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator(weights_test);

	std::osyncstream(std::cout) << "nmiReLU-JN: Training... " << '\n';

	std::atomic_flag data_is_ready_for_workers;
	std::atomic_uint workers_is_ready_for_data;
	bool threads_run = true;

	bool learn_nonmonotonic = true;
	bool calc_loss = false;
	std::atomic<float> threads_loss;

	auto TrainWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount());
		nn::errcalc::ErrorCalcSoftMAX softmax_calculator(10);

		float loss;
		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			loss = 0.0f;
			// Learning
			learnguider_mtr_nm.WorkerDoForward(worker_id, &caches[0]);
			loss += learnguider_mtr_nm.WorkerFillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
			learnguider_mtr_nm.WorkerDoBackward(worker_id, &caches[0]);

			// Loss & local accuracy
			if (calc_loss) {
				f1score_train_calculator.AppendResultsThreadSafe(learnguider_mtr_nm.GetOutputs(), perfect_out_store, worker_id, THREADS_COUNT);
				threads_loss.fetch_add(loss, std::memory_order_relaxed);
			}

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	std::vector<std::thread> workers;
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TrainWorkerThread, i);
	}

	unsigned current_threads_count;
	unsigned log_iterations = 0;
	unsigned batch_iterations = 0;
	unsigned batch_iterations_all = mnist_train.size() / BATCH_SIZE;
	for (unsigned epoch = 0; epoch != EPOCHS; ++epoch) {
		std::shuffle(mnist_train.begin(), mnist_train.end(), preudorandom);

		auto datase_current_pos = mnist_train.cbegin();
		auto datase_end = mnist_train.cend();

		batch_iterations = 0;
		// Training epoch
		do {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				auto &out = perfect_out_store[batch_i];
				std::fill(out.begin(), out.end(), 0.0f);
				out[datase_current_pos->label] = 1.0f;
			}
			++batch_iterations;

			calc_loss = false;
			if (++log_iterations == LOG_INTERVAL) {
				log_iterations = 0;
				calc_loss = true;
			}

			data_is_ready_for_workers.test_and_set(std::memory_order_release);
			data_is_ready_for_workers.notify_all();

			// Training is performing now

			current_threads_count = 0;
			do {
				workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
				current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
			} while (current_threads_count != THREADS_COUNT);
			workers_is_ready_for_data.store(0, std::memory_order_release);

			if (calc_loss) {
				float loss = threads_loss.load(std::memory_order_relaxed);
				threads_loss.store(0.0f, std::memory_order_relaxed);
				loss /= BATCH_SIZE;
				std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nmiReLU-JN: Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << f1score_train_calculator.CalcAccuracy() << "%  F1(local): " << f1score_train_calculator.CalcF1() << "\n";
				f1score_train_calculator.Reset();
			}
		} while (datase_current_pos != datase_end);

		// Fast validation after epoch
		datase_current_pos = mnist_valid.cbegin();
		datase_end = mnist_valid.cend();
		while (true) {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				perfect_outs[batch_i] = datase_current_pos->label;
			}
			if (datase_current_pos == datase_end) {
				break;
			}

			learnguider_sing_nm.DoForward();
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
				f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_sing_nm.GetOutputs(), batch_i), perfect_outs[batch_i]);
			}
		}
		std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nmiReLU-JN: " << (learn_nonmonotonic ? "<NON-MONOTONIC>" : "<EQUIVALENT>") << "\t\t\tAccuracy (validation dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
		f1score_test_calculator.Reset();
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();
	// Ensuring all released
	for (auto &thr : workers) {
		thr.join();
	}

	// Non-monotonity usage stats: minus_count, plus_count, min, summ, max
	std::map<const nnNmReLU *, std::tuple<unsigned, unsigned, unsigned, unsigned>> fu_both_neurons;

	auto NonMonotonityStatProc = [&](nnNmReLU *nrn) {
		auto iter = fu_both_neurons.find(nrn);
		if (iter != fu_both_neurons.end()) {
			float accval;
			for (unsigned batch_i = 0; batch_i != nrn->GetCurrentBatchSize(); ++batch_i) {
				accval = nrn->RealAccumulatorValue(batch_i);
				if (accval < 0) {
					++std::get<0>(iter->second);
					if (accval < -NM1_MAX_VALUE) {
						++std::get<1>(iter->second);
					}
				} else {
					++std::get<2>(iter->second);
					if (nrn->RealAccumulatorValue(batch_i) > NM1_MAX_VALUE) {
						++std::get<3>(iter->second);
					}
				}
			}
		} else {
			float accval;
			{
				accval = nrn->RealAccumulatorValue();
				fu_both_neurons.emplace(nrn, std::make_tuple((unsigned)(accval < 0), (unsigned)(accval < -NM1_MAX_VALUE), (unsigned)(accval >= 0), (unsigned)(accval > NM1_MAX_VALUE)));
			}
			iter = fu_both_neurons.find(nrn);
			for (unsigned batch_i = 1; batch_i < nrn->GetCurrentBatchSize(); ++batch_i) {
				accval = nrn->RealAccumulatorValue(batch_i);
				if (accval < 0) {
					++std::get<0>(iter->second);
					if (accval < -NM1_MAX_VALUE) {
						++std::get<1>(iter->second);
					}
				} else {
					++std::get<2>(iter->second);
					if (nrn->RealAccumulatorValue(batch_i) > NM1_MAX_VALUE) {
						++std::get<3>(iter->second);
					}
				}
			}
		}
	};


	std::osyncstream(std::cout) << "nmiReLU-JN: Accuracy calculating...\n";

	// Accuracy calc
	auto TestWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount(false));
		std::vector<float> *caches_ptr = caches.size() ? &caches[0] : nullptr;

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			learnguider_mtr_nm.WorkerDoForward(worker_id, caches_ptr);
			f1score_test_calculator.AppendResultsThreadSafe(learnguider_mtr_nm.GetOutputs(), perfect_outs, worker_id, THREADS_COUNT);

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	// Threads prepare
	data_is_ready_for_workers.clear();
	threads_run = true;

	workers.clear();
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TestWorkerThread, i);
	}

	// Accuracy calc
	unsigned batch_count = 0;
	auto test_current_pos = mnist_test.cbegin();
	auto test_end = mnist_test.cend();
	batch_iterations_all = mnist_test.size() / BATCH_SIZE;

	log_iterations = 0;
	while (true) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++test_current_pos) {
			if (test_current_pos == test_end) {
				break;
			}
			auto iter = inputs_store[batch_i].begin();
			for (auto item : test_current_pos->input) {
				*iter = UINT8_NORM * item;
				++iter;
			}
			perfect_outs[batch_i] = test_current_pos->label;
		}
		if (test_current_pos == test_end) {
			break;
		}
		++batch_count;
		if (++log_iterations == LOG_INTERVAL) {
			log_iterations = 0;
			std::osyncstream(std::cout) << "nmiReLU-JN: Batch: " << batch_count << '/' << batch_iterations_all << '\n';
		}

		data_is_ready_for_workers.test_and_set(std::memory_order_release);
		data_is_ready_for_workers.notify_all();

		// Evaluating is performing now

		current_threads_count = 0;
		do {
			workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
			current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
		} while (current_threads_count != THREADS_COUNT);
		workers_is_ready_for_data.store(0, std::memory_order_release);

		// Grab non-monotonity stat
		for (auto lr : learnguider_mtr_nm.GetLayers()) {
			for (auto nrn : lr->Neurons()) {
				NonMonotonityStatProc(dynamic_cast<nnNmReLU *>(nrn));
			}
		}
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();

	float stand_percent = 0.0f;
	float extra_percent = 0.0f;
	for (auto &iter : fu_both_neurons) {
		unsigned smin = std::min(std::get<0>(iter.second), std::get<2>(iter.second));
		unsigned smax = std::max(std::get<0>(iter.second), std::get<2>(iter.second));
		unsigned emin = std::min(std::get<1>(iter.second), std::get<3>(iter.second));
		unsigned emax = std::max(std::get<1>(iter.second), std::get<3>(iter.second));

		stand_percent += static_cast<float>(smin) / smax;
		extra_percent += static_cast<float>(emin) / emax;

	}
	stand_percent *= 100.0f / fu_both_neurons.size();
	extra_percent *= 100.0f / fu_both_neurons.size();

	float accuracy = f1score_test_calculator.CalcAccuracy();
	float f1 = f1score_test_calculator.CalcF1();
	std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nmiReLU-JN: Accuracy (test dataset): " << accuracy << "%  F1: " << f1 << "  Both-f-usg:: stand: " << stand_percent << "%  extra: " << extra_percent << "%\n";

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}

	std::vector<float> weights_store;
	for (auto ifc : weights_iter) {
		weights_store.push_back(ifc->Weight());
	}

	std::uniform_int_distribution<unsigned> randistributor_seed(0, std::numeric_limits<unsigned>::max());

	auto attack_success_rate = Attack_nmiReLU(randistributor_seed(preudorandom), true, false, weights_store);

	return MNIST_Attack_nmiReLU_result{ accuracy, f1, stand_percent, extra_percent, attack_success_rate.cwl2_success_rate, attack_success_rate.bim_success_rate, attack_success_rate.cwl2_iters, attack_success_rate.bim_iters };
}

static inline MNIST_Attack_nmiReLU_result MNIST_Attack_nmiReLU_nm0(unsigned random_seed, const std::vector<MNIST_entry> &mnist_train_sample, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test) {
	std::mt19937 preudorandom(random_seed);

	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);
	std::uniform_real_distribution<float> randistributor_diviation(-0.0025f, 0.0025f);

	std::vector<MNIST_entry> mnist_train = mnist_train_sample;

	using nnNmReLU = nn::NNB_nm1ReLUb<BATCH_SIZE, true, true>; // Non-monotonic ReLU, using Kahan summation
	using nnLinSumm = nn::NNB_LinearSlimT<BATCH_SIZE, true>; // Linear summator, using Kahan

	// Input image (like vector, flatten)
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(BATCH_SIZE);

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<BATCH_SIZE>> layer_inp(INPUT_SIZE, [&](nn::NNB_InputB<BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Non-monotonic Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer1_nm_hid(NM_LAYER_HIDDEN1_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Hidden layer 2
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer2_nm_hid(LAYER_HIDDEN2_SIZE, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});
	// Non-monotonic Output layer
	nn::NeuronHoldingStaticLayer<nnNmReLU> layer3_nm_out(10, [&](nnNmReLU *const mem_ptr, unsigned) {
		new(mem_ptr)nnNmReLU(NM1_MAX_VALUE);
	});

	using OptimAlg = nn::optimizers::Adam;
	OptimAlg optimizer(LEARNING_RATE);

	// Connections of Non-Monotinic (weights initialization is not required if equivalent trained first)
	// Bias
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid1(&layer_bias, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmhid2(&layer_bias, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connections_bias_nmout(&layer_bias, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, -NM1_MAX_VALUE + randistributor_diviation(preudorandom));
	});
	// Non-Monotinic FeedForward connections
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_inp_nmhid1(&layer_inp, &layer1_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid1_nmhid2(&layer1_nm_hid, &layer2_nm_hid, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});
	nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>> connection_nmhid2_nmout(&layer2_nm_hid, &layer3_nm_out, [&](nn::NNB_Connection<OptimAlg> *const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to) {
		new(mem_ptr)nn::NNB_Connection<OptimAlg>(from, to, &optimizer, randistributor(preudorandom));
	});

	// Combining connectoms into one structure
	using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<OptimAlg>>::CBIterator;
	IterableAggregation<nn::interfaces::CBI *> weights_iter;
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid1);
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmhid2);
	weights_iter.AppendIterableItem<IterType>(connections_bias_nmout);
	weights_iter.AppendIterableItem<IterType>(connection_inp_nmhid1);
	weights_iter.AppendIterableItem<IterType>(connection_nmhid1_nmhid2);
	weights_iter.AppendIterableItem<IterType>(connection_nmhid2_nmout);


	nn::LearnGuiderFwBPgThreadAble learnguider_mtr_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }, BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training

	nn::LearnGuiderFwBPg learnguider_sing_nm({ &layer1_nm_hid, &layer2_nm_hid, &layer3_nm_out }); // Evaluator for validating & testing

	const float UINT8_NORM = 1.0f / 255.0f;

	// Learning
	std::vector<std::vector<float>> perfect_out_store(BATCH_SIZE, std::vector<float>(10));
	std::vector<unsigned short> perfect_outs(BATCH_SIZE);

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_train_calculator(weights_train);
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator(weights_test);

	std::osyncstream(std::cout) << "nm0ReLU: Training... " << '\n';

	std::atomic_flag data_is_ready_for_workers;
	std::atomic_uint workers_is_ready_for_data;
	bool threads_run = true;

	bool learn_nonmonotonic = true;
	bool calc_loss = false;
	std::atomic<float> threads_loss;

	auto TrainWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount());
		nn::errcalc::ErrorCalcSoftMAX softmax_calculator(10);

		float loss;
		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			loss = 0.0f;
			// Learning
			learnguider_mtr_nm.WorkerDoForward(worker_id, &caches[0]);
			loss += learnguider_mtr_nm.WorkerFillupOutsError(worker_id, &softmax_calculator, perfect_out_store, calc_loss);
			learnguider_mtr_nm.WorkerDoBackward(worker_id, &caches[0]);

			// Loss & local accuracy
			if (calc_loss) {
				f1score_train_calculator.AppendResultsThreadSafe(learnguider_mtr_nm.GetOutputs(), perfect_out_store, worker_id, THREADS_COUNT);
				threads_loss.fetch_add(loss, std::memory_order_relaxed);
			}

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	std::vector<std::thread> workers;
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TrainWorkerThread, i);
	}

	unsigned current_threads_count;
	unsigned log_iterations = 0;
	unsigned batch_iterations = 0;
	unsigned batch_iterations_all = mnist_train.size() / BATCH_SIZE;
	for (unsigned epoch = 0; epoch != EPOCHS; ++epoch) {
		std::shuffle(mnist_train.begin(), mnist_train.end(), preudorandom);

		auto datase_current_pos = mnist_train.cbegin();
		auto datase_end = mnist_train.cend();

		batch_iterations = 0;
		// Training epoch
		do {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				auto &out = perfect_out_store[batch_i];
				std::fill(out.begin(), out.end(), 0.0f);
				out[datase_current_pos->label] = 1.0f;
			}
			++batch_iterations;

			calc_loss = false;
			if (++log_iterations == LOG_INTERVAL) {
				log_iterations = 0;
				calc_loss = true;
			}

			data_is_ready_for_workers.test_and_set(std::memory_order_release);
			data_is_ready_for_workers.notify_all();

			// Training is performing now

			current_threads_count = 0;
			do {
				workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
				current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
			} while (current_threads_count != THREADS_COUNT);
			workers_is_ready_for_data.store(0, std::memory_order_release);

			if (calc_loss) {
				float loss = threads_loss.load(std::memory_order_relaxed);
				threads_loss.store(0.0f, std::memory_order_relaxed);
				loss /= BATCH_SIZE;
				std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nm0ReLU: Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << f1score_train_calculator.CalcAccuracy() << "%  F1(local): " << f1score_train_calculator.CalcF1() << "\n";
				f1score_train_calculator.Reset();
			}
		} while (datase_current_pos != datase_end);

		// Fast validation after epoch
		datase_current_pos = mnist_valid.cbegin();
		datase_end = mnist_valid.cend();
		while (true) {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++datase_current_pos) {
				if (datase_current_pos == datase_end) {
					break;
				}
				auto iter = inputs_store[batch_i].begin();
				for (auto item : datase_current_pos->input) {
					*iter = UINT8_NORM * item;
					++iter;
				}
				perfect_outs[batch_i] = datase_current_pos->label;
			}
			if (datase_current_pos == datase_end) {
				break;
			}

			learnguider_sing_nm.DoForward();
			for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i) {
				f1score_test_calculator.AppendResult(nn::netquality::NeuroArgmax(learnguider_sing_nm.GetOutputs(), batch_i), perfect_outs[batch_i]);
			}
		}
		std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nm0ReLU: " << (learn_nonmonotonic ? "<NON-MONOTONIC>" : "<EQUIVALENT>") << "\t\t\tAccuracy (validation dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
		f1score_test_calculator.Reset();
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();
	// Ensuring all released
	for (auto &thr : workers) {
		thr.join();
	}

	// Non-monotonity usage stats: minus_count, plus_count, min, summ, max
	std::map<const nnNmReLU *, std::tuple<unsigned, unsigned, unsigned, unsigned>> fu_both_neurons;

	auto NonMonotonityStatProc = [&](nnNmReLU *nrn) {
		auto iter = fu_both_neurons.find(nrn);
		if (iter != fu_both_neurons.end()) {
			float accval;
			for (unsigned batch_i = 0; batch_i != nrn->GetCurrentBatchSize(); ++batch_i) {
				accval = nrn->RealAccumulatorValue(batch_i);
				if (accval < 0) {
					++std::get<0>(iter->second);
					if (accval < -NM1_MAX_VALUE) {
						++std::get<1>(iter->second);
					}
				} else {
					++std::get<2>(iter->second);
					if (nrn->RealAccumulatorValue(batch_i) > NM1_MAX_VALUE) {
						++std::get<3>(iter->second);
					}
				}
			}
		} else {
			float accval;
			{
				accval = nrn->RealAccumulatorValue();
				fu_both_neurons.emplace(nrn, std::make_tuple((unsigned)(accval < 0), (unsigned)(accval < -NM1_MAX_VALUE), (unsigned)(accval >= 0), (unsigned)(accval > NM1_MAX_VALUE)));
			}
			iter = fu_both_neurons.find(nrn);
			for (unsigned batch_i = 1; batch_i < nrn->GetCurrentBatchSize(); ++batch_i) {
				accval = nrn->RealAccumulatorValue(batch_i);
				if (accval < 0) {
					++std::get<0>(iter->second);
					if (accval < -NM1_MAX_VALUE) {
						++std::get<1>(iter->second);
					}
				} else {
					++std::get<2>(iter->second);
					if (nrn->RealAccumulatorValue(batch_i) > NM1_MAX_VALUE) {
						++std::get<3>(iter->second);
					}
				}
			}
		}
	};


	std::osyncstream(std::cout) << "nm0ReLU: Accuracy calculating...\n";

	// Accuracy calc
	auto TestWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(learnguider_mtr_nm.GetRequiredCachesCount(false));
		std::vector<float> *caches_ptr = caches.size() ? &caches[0] : nullptr;

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			learnguider_mtr_nm.WorkerDoForward(worker_id, caches_ptr);
			f1score_test_calculator.AppendResultsThreadSafe(learnguider_mtr_nm.GetOutputs(), perfect_outs, worker_id, THREADS_COUNT);

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};

	// Threads prepare
	data_is_ready_for_workers.clear();
	threads_run = true;

	workers.clear();
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TestWorkerThread, i);
	}

	// Accuracy calc
	unsigned batch_count = 0;
	auto test_current_pos = mnist_test.cbegin();
	auto test_end = mnist_test.cend();
	batch_iterations_all = mnist_test.size() / BATCH_SIZE;

	log_iterations = 0;
	while (true) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != BATCH_SIZE; ++batch_i, ++test_current_pos) {
			if (test_current_pos == test_end) {
				break;
			}
			auto iter = inputs_store[batch_i].begin();
			for (auto item : test_current_pos->input) {
				*iter = UINT8_NORM * item;
				++iter;
			}
			perfect_outs[batch_i] = test_current_pos->label;
		}
		if (test_current_pos == test_end) {
			break;
		}
		++batch_count;
		if (++log_iterations == LOG_INTERVAL) {
			log_iterations = 0;
			std::osyncstream(std::cout) << "nm0ReLU: Batch: " << batch_count << '/' << batch_iterations_all << '\n';
		}

		data_is_ready_for_workers.test_and_set(std::memory_order_release);
		data_is_ready_for_workers.notify_all();

		// Evaluating is performing now

		current_threads_count = 0;
		do {
			workers_is_ready_for_data.wait(current_threads_count, std::memory_order_relaxed);
			current_threads_count = workers_is_ready_for_data.load(std::memory_order_acquire);
		} while (current_threads_count != THREADS_COUNT);
		workers_is_ready_for_data.store(0, std::memory_order_release);

		// Grab non-monotonity stat
		for (auto lr : learnguider_mtr_nm.GetLayers()) {
			for (auto nrn : lr->Neurons()) {
				NonMonotonityStatProc(dynamic_cast<nnNmReLU *>(nrn));
			}
		}
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();

	float stand_percent = 0.0f;
	float extra_percent = 0.0f;
	for (auto &iter : fu_both_neurons) {
		unsigned smin = std::min(std::get<0>(iter.second), std::get<2>(iter.second));
		unsigned smax = std::max(std::get<0>(iter.second), std::get<2>(iter.second));
		unsigned emin = std::min(std::get<1>(iter.second), std::get<3>(iter.second));
		unsigned emax = std::max(std::get<1>(iter.second), std::get<3>(iter.second));

		stand_percent += static_cast<float>(smin) / smax;
		extra_percent += static_cast<float>(emin) / emax;

	}
	stand_percent *= 100.0f / fu_both_neurons.size();
	extra_percent *= 100.0f / fu_both_neurons.size();

	float accuracy = f1score_test_calculator.CalcAccuracy();
	float f1 = f1score_test_calculator.CalcF1();
	std::osyncstream(std::cout) << std::fixed << std::setprecision(3) << "nm0ReLU: Accuracy (test dataset): " << accuracy << "%  F1: " << f1 << "  Both-f-usg:: stand: " << stand_percent << "%  extra: " << extra_percent << "%\n";

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}

	std::vector<float> weights_store;
	for (auto ifc : weights_iter) {
		weights_store.push_back(ifc->Weight());
	}

	std::uniform_int_distribution<unsigned> randistributor_seed(0, std::numeric_limits<unsigned>::max());

	auto attack_success_rate = Attack_nmiReLU(randistributor_seed(preudorandom), false, false, weights_store);

	return MNIST_Attack_nmiReLU_result{ accuracy, f1, stand_percent, extra_percent, attack_success_rate.cwl2_success_rate, attack_success_rate.bim_success_rate, attack_success_rate.cwl2_iters, attack_success_rate.bim_iters };
}

static void CWL2_ReLU_StatePrinter(unsigned short binstep, unsigned iteration, float loss) {
	if (iteration == std::numeric_limits<unsigned>::max()) {
		std::osyncstream(std::cout) << "ReLU:CWL2:: done binSearch: " << binstep << "  final loss: " << loss << '\n';
	} else {
		std::osyncstream(std::cout) << "ReLU:CWL2:: binSearch: " << binstep << "  iteration: " << iteration << "  loss: " << loss << '\n';
	}
};

static void BIM_ReLU_StatePrinter(unsigned iteration, float loss) {
	std::osyncstream(std::cout) << "ReLU:BIM::  iteration: " << iteration << "  loss: " << loss << '\n';
};

static void CWL2_nmiReLUeq_StatePrinter(unsigned short binstep, unsigned iteration, float loss) {
	if (iteration == std::numeric_limits<unsigned>::max()) {
		std::osyncstream(std::cout) << "nmiReLU-EQ:CWL2:: done binSearch: " << binstep << "  final loss: " << loss << '\n';
	} else {
		std::osyncstream(std::cout) << "nmiReLU-EQ:CWL2:: binSearch: " << binstep << "  iteration: " << iteration << "  loss: " << loss << '\n';
	}
};

static void BIM_nmiReLUeq_StatePrinter(unsigned iteration, float loss) {
	std::osyncstream(std::cout) << "nmiReLU-EQ:BIM::  iteration: " << iteration << "  loss: " << loss << '\n';
};

static void CWL2_nmiReLUjn_StatePrinter(unsigned short binstep, unsigned iteration, float loss) {
	if (iteration == std::numeric_limits<unsigned>::max()) {
		std::osyncstream(std::cout) << "nmiReLU-JN:CWL2:: done binSearch: " << binstep << "  final loss: " << loss << '\n';
	} else {
		std::osyncstream(std::cout) << "nmiReLU-JN:CWL2:: binSearch: " << binstep << "  iteration: " << iteration << "  loss: " << loss << '\n';
	}
};

static void BIM_nmiReLUjn_StatePrinter(unsigned iteration, float loss) {
	std::osyncstream(std::cout) << "nmiReLU-JN:BIM::  iteration: " << iteration << "  loss: " << loss << '\n';
};

static void CWL2_nm0ReLU_StatePrinter(unsigned short binstep, unsigned iteration, float loss) {
	if (iteration == std::numeric_limits<unsigned>::max()) {
		std::osyncstream(std::cout) << "nm0ReLU:CWL2:: done binSearch: " << binstep << "  final loss: " << loss << '\n';
	} else {
		std::osyncstream(std::cout) << "nm0ReLU:CWL2:: binSearch: " << binstep << "  iteration: " << iteration << "  loss: " << loss << '\n';
	}
};

static void BIM_nm0ReLU_StatePrinter(unsigned iteration, float loss) {
	std::osyncstream(std::cout) << "nm0ReLU:BIM::  iteration: " << iteration << "  loss: " << loss << '\n';
};