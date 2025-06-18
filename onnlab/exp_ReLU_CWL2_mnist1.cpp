#include "onnlab.h"

#include "NNB_Connection_spyable.h"
#include "OptimizerAdam.h"
#include "OptimizerGD.h"
#include "NNB_ReLU.h"
#include "NNB_ConstInput.h"
#include "NNB_Input_spyable.h"
#include "NNB_Layer.h"
#include "LearnGuiderFwBPg.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"
#include "IterableAggregation.h"

#include "CarliniWagnerL2.h"

#include "CSVreader.h"

#include <vector>
#include <random>
#include <array>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

static const unsigned THREADS_COUNT = 32;
static const unsigned LOG_INTERVAL = 200;

static const unsigned ATTACK_BATCH_SIZE = 8;
static const unsigned LEARNING_BATCH_SIZE = 64;
static const unsigned LEARNING_EPOCHS = 10;

static const float LEARNING_RATE = 0.001f;

static const unsigned LAYER_HIDDEN1_SIZE = 64;
static const unsigned LAYER_HIDDEN2_SIZE = 16;

static const unsigned RANDOM_SEED = 42;

struct MNIST_entry {
	MNIST_entry(uint8_t label, std::vector<uint8_t> input, uint16_t reseve):label(label), input(input) {
		input.reserve(reseve);
	}
	uint8_t label;
	std::vector<uint8_t> input;
};

static void Learn_ReLU_mnist(const unsigned RANDOM_SEED_C, const std::vector<MNIST_entry> &mnist_test, const nn::netquality::ClassWeightsCalculator &mnist_test_classes_w);
static bool Validate_ReLU_mnist(nn::LearnGuiderFwBPgThreadAble &learnguider, float ref_accuracy, float ref_f1score, const std::vector<MNIST_entry> &mnist_test, const nn::netquality::ClassWeightsCalculator &mnist_test_classes_w);
static std::vector<float> Infer_ReLU_mnist(nn::LearnGuiderFwBPgThreadAble &inferguider, std::vector<float> &inputs);
static void CWL2StatePrinter(unsigned short binstep, unsigned iteration, float loss);

void exp_ReLU_CWL2_mnist1() {
	std::cout << "exp_ReLU_CWL2_mnist1" << std::endl;

	std::mt19937 preudorandom(RANDOM_SEED);
	std::uniform_int_distribution<unsigned> seed_gen(0, std::numeric_limits<unsigned>::max());

	std::cout << "Loading test dataset...\n";
	std::vector<MNIST_entry> mnist_test;
	{
		CSVreader csv("../mnist-in-csv/mnist_test.csv");
		auto rowsz = csv.FetchNextRow();
		while (rowsz) {
			mnist_test.emplace_back(atoi(csv[0]), std::vector<uint8_t>(), rowsz - 1); // label writing
			for (unsigned i = 1; i < rowsz; ++i) 
				mnist_test.back().input.push_back(atoi(csv[i]));
			rowsz = csv.FetchNextRow();
		}
	}
	nn::netquality::ClassWeightsCalculator mnist_test_classes_w(10);
	for (auto &itm : mnist_test) {
		mnist_test_classes_w.NoteSample(itm.label);
	}
	mnist_test_classes_w.CalcWeights();
	mnist_test_classes_w.KeepWeightsOnly();

	if (!std::filesystem::exists("exp_ReLU_CWL2_mnist1_WEIGHTS.bin")) {
		std::cout << "Weights file not found, training new weights...\n";
		Learn_ReLU_mnist(seed_gen(preudorandom), mnist_test, mnist_test_classes_w);
	}

	// BEGIN MODEL DEFINE
	const unsigned INPUT_SIZE = 28 * 28;
	using nnReLU = nn::NNB_ReLUb<ATTACK_BATCH_SIZE, true>; // Using Kahan summation
	using nnInput = nn::NNB_Input_spyableB<ATTACK_BATCH_SIZE, true>; // Using Kahan summation
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(ATTACK_BATCH_SIZE);
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) { new(mem_ptr)nn::NNB_ConstInput; });
	nn::NeuronHoldingStaticLayer<nnInput> layer_inp(INPUT_SIZE, [&](nnInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nnInput([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity; for (size_t i = 0; i != capacity; ++i) storage[i] = &inputs_store[i][index];
		});
	});
	nn::NeuronHoldingStaticLayer<nnReLU> layer_hid1(LAYER_HIDDEN1_SIZE, [&](nnReLU *const mem_ptr, unsigned) { new(mem_ptr)nnReLU; });
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


	float ref_accuracy, ref_f1score;
	{ // Restoring weights
		using IterType = nn::DenseLayerStaticConnectomHolder<nn::NNB_Connection<NoOptim>>::CBIterator;
		IterableAggregation<nn::interfaces::CBI *> weights_iter;
		weights_iter.AppendIterableItem<IterType>(connection_bias_hid1);
		weights_iter.AppendIterableItem<IterType>(connection_bias_hid2);
		weights_iter.AppendIterableItem<IterType>(connection_bias_out);
		weights_iter.AppendIterableItem<IterType>(connection_inp_hid1);
		weights_iter.AppendIterableItem<IterType>(connection_hid1_hid2);
		weights_iter.AppendIterableItem<IterType>(connection_hid2_out);

		std::ifstream weights_store("exp_ReLU_CWL2_mnist1_WEIGHTS.bin", std::ios::binary);
		weights_store.read(reinterpret_cast<char *>(&ref_accuracy), sizeof(float));
		weights_store.read(reinterpret_cast<char *>(&ref_f1score), sizeof(float));
		float weight;
		for (auto ifc : weights_iter) {
			weights_store.read(reinterpret_cast<char *>(&weight), sizeof(float));
			ifc->Weight(weight);
		}
		weights_store.close();
	}

	if (Validate_ReLU_mnist(inferguider, ref_accuracy, ref_f1score, mnist_test, mnist_test_classes_w)) {
		std::cout << "Ready.\n";
	} else {
		std::cout << "Error!\n";
		return;
	}

	std::uniform_real_distribution<float> randistributor(0.3f, 0.7f);
	std::uniform_int_distribution<unsigned short> randistributor_cls(0, 9);

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

	{
		std::cout << "Running CWL2...\n";

		nn::reverse::CarliniWagnerL2Params cwl2params;
		cwl2params.box_min = 0.0f;
		cwl2params.box_max = 1.0f;
		cwl2params.early_stop_chances_count = 16;
		cwl2params.state_printing_period = 10;
		cwl2params.StatePrinter = &CWL2StatePrinter;
		//cwl2params.allow_early_stop = false;

		nn::reverse::CarliniWagnerL2ThreadAble cwl2(inferguider, cwl2params, ATTACK_BATCH_SIZE, THREADS_COUNT);

		auto attack_results = cwl2.RunAttack(sources, targets);

		std::cout << "Results:\n";

		std::cout << std::fixed << std::setprecision(3);

		auto tit = targets.begin();
		for (auto &attk : attack_results) {
			if (attk.size()) {
				std::cout << "Success, class: " << *tit << "\n\n";
			} else {
				std::cout << "[!] Failed, target class: " << *tit << "\n\n";
			}
			++tit;
		}
	}

	return;
}

static void CWL2StatePrinter(unsigned short binstep, unsigned iteration, float loss) {
	if (iteration == std::numeric_limits<unsigned>::max()) {
		std::cout << "CWL2:: done binSearch: " << binstep << "  final loss: " << loss << '\n';
	} else {
		std::cout << "CWL2:: binSearch: " << binstep << "  iteration: " << iteration << "  loss: " << loss << '\n';
	}
};

static std::vector<float> Infer_ReLU_mnist(nn::LearnGuiderFwBPgThreadAble &inferguider, std::vector<float> &inputs) {
	std::vector<float> outputs(11);

	auto &inputsnr = inferguider.GetLayers()[0]->Neurons();


	return outputs;
}

static bool Validate_ReLU_mnist(nn::LearnGuiderFwBPgThreadAble &inferguider, float ref_accuracy, float ref_f1score, const std::vector<MNIST_entry> &mnist_test, const nn::netquality::ClassWeightsCalculator &mnist_test_classes_w) {
	std::cout << "Model validation...\n";

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator(mnist_test_classes_w.GetWeights());

	std::vector<unsigned short> perfect_outs(ATTACK_BATCH_SIZE);
	const float UINT8_NORM = 1.0f / 255.0f;

	std::atomic_flag data_is_ready_for_workers;
	std::atomic_uint workers_is_ready_for_data;
	bool threads_run = true;

	auto TestWorkerThread = [&](unsigned worker_id) {
		std::vector<std::vector<float>> caches(inferguider.GetRequiredCachesCount(false));
		std::vector<float> *caches_ptr = caches.size() ? &caches[0] : nullptr;

		while (threads_run) {
			data_is_ready_for_workers.wait(false, std::memory_order_acquire);

			if (!threads_run)
				break;

			inferguider.WorkerDoForward(worker_id, caches_ptr);
			f1score_test_calculator.AppendResultsThreadSafe(inferguider.GetOutputs(), perfect_outs, worker_id, THREADS_COUNT);

			data_is_ready_for_workers.clear(std::memory_order_release);

			if (workers_is_ready_for_data.fetch_add(1, std::memory_order_release) + 1 == THREADS_COUNT) {
				workers_is_ready_for_data.notify_one();
			}
		}
	};
	
	// Threads prepare
	std::vector<std::thread> workers;
	for (unsigned i = 0; i != THREADS_COUNT; ++i) {
		workers.emplace_back(TestWorkerThread, i);
	}

	unsigned batch_count = 0;
	auto test_current_pos = mnist_test.cbegin();
	auto test_end = mnist_test.cend();
	unsigned batch_iterations_all = mnist_test.size() / ATTACK_BATCH_SIZE;

	auto &inputs = inferguider.GetLayers()[0]->Neurons();

	unsigned log_iterations = 0, current_threads_count;
	while (true) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != ATTACK_BATCH_SIZE; ++batch_i, ++test_current_pos) {
			if (test_current_pos == test_end) {
				break;
			}
			auto iter = inputs.begin();
			for (auto item : test_current_pos->input) {
				dynamic_cast<nn::interfaces::InputNeuronI *>(*iter)->SetOwnLevel(UINT8_NORM * item, batch_i);
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
			std::cout << "Model validation: " << batch_count << '/' << batch_iterations_all << '\n';
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

	std::cout << "Model validation resluts (test dataset): " << accuracy << "%  F1: " << f1 << "\n";

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}

	return std::abs(ref_accuracy - accuracy) < 1e-2f && std::abs(ref_f1score - f1) < 1e-4f;
}

static void Learn_ReLU_mnist(const unsigned RANDOM_SEED_C, const std::vector<MNIST_entry> &mnist_test, const nn::netquality::ClassWeightsCalculator &mnist_test_classes_w) {
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

	std::mt19937 preudorandom(RANDOM_SEED_C);

	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);

	const unsigned INPUT_SIZE = 28 * 28;

	using nnReLU = nn::NNB_ReLUb<LEARNING_BATCH_SIZE, true>; // Using Kahan summation

	// Input image (like vector, flatten)
	std::vector<std::array<float, INPUT_SIZE>> inputs_store(LEARNING_BATCH_SIZE);

	// Bias layer
	nn::NeuronHoldingStaticLayer<nn::NNB_ConstInput> layer_bias(1, [&](nn::NNB_ConstInput *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_ConstInput;
	});

	// Input layer
	nn::NeuronHoldingStaticLayer<nn::NNB_InputB<LEARNING_BATCH_SIZE>> layer_inp(INPUT_SIZE, [&](nn::NNB_InputB<LEARNING_BATCH_SIZE> *const mem_ptr, unsigned index) {
		new(mem_ptr)nn::NNB_InputB<LEARNING_BATCH_SIZE>([&](float **storage, unsigned capacity, unsigned &count) {
			count = capacity;
			for (size_t i = 0; i != capacity; ++i) {
				storage[i] = &inputs_store[i][index];
			}
		});
	});

	// Hidden layer 1
	nn::NeuronHoldingStaticLayer<nnReLU> layer_hid1(LAYER_HIDDEN1_SIZE, [&](nnReLU *const mem_ptr, unsigned) {
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

	nn::LearnGuiderFwBPgThreadAble learnguider_multhr({ &layer_hid1, &layer_hid2, &layer_out }, LEARNING_BATCH_SIZE, THREADS_COUNT); // Evaluator & backpropagator for training

	const float UINT8_NORM = 1.0f / 255.0f;

	// Learning
	std::vector<std::vector<float>> perfect_out_store(LEARNING_BATCH_SIZE, std::vector<float>(10));

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_train_calculator(mnist_train_classes_w.GetWeights());
	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator(mnist_test_classes_w.GetWeights());

	std::cout << std::fixed << std::setprecision(3);

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

	unsigned current_threads_count;
	unsigned log_iterations = 0;
	unsigned batch_iterations = 0;
	unsigned batch_iterations_all = mnist_train.size() / LEARNING_BATCH_SIZE;
	for (unsigned epoch = 0; epoch != LEARNING_EPOCHS; ++epoch) {
		std::shuffle(mnist_train.begin(), mnist_train.end(), preudorandom);

		auto datase_current_pos = mnist_train.cbegin();
		auto datase_end = mnist_train.cend();

		batch_iterations = 0;
		// Training epoch
		do {
			// Update inputs
			for (unsigned batch_i = 0; batch_i != LEARNING_BATCH_SIZE; ++batch_i, ++datase_current_pos) {
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
				loss /= LEARNING_BATCH_SIZE;
				std::cout << "Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << accuracy << "%  F1(local): " << f1 << "\n";
			}
		} while (datase_current_pos != datase_end);
	}

	// Releasing threads
	threads_run = false;
	data_is_ready_for_workers.test_and_set(std::memory_order_relaxed);
	data_is_ready_for_workers.notify_all();
	// Ensuring all released
	for (auto &thr : workers) {
		thr.join();
	}

	std::cout << "Accuracy calculating...\n";

	std::vector<unsigned short> perfect_outs(LEARNING_BATCH_SIZE);

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
	batch_iterations_all = mnist_test.size() / LEARNING_BATCH_SIZE;

	nn::netquality::F1scoreMulticlassWeightsGlobal f1score_test_calculator000(mnist_test_classes_w.GetWeights());

	log_iterations = 0;
	while (true) {
		// Update inputs
		for (unsigned batch_i = 0; batch_i != LEARNING_BATCH_SIZE; ++batch_i, ++test_current_pos) {
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
			std::cout << "Batch: " << batch_count << '/' << batch_iterations_all << '\n';
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

	std::cout << "Accuracy (test dataset): " << accuracy << "%  F1: " << f1 << "\n";

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}

	std::ofstream weights_store("exp_ReLU_CWL2_mnist1_WEIGHTS.bin", std::ios::binary);

	weights_store.write(reinterpret_cast<char *>(&accuracy), sizeof(float));
	weights_store.write(reinterpret_cast<char *>(&f1), sizeof(float));
	float weight;
	for (auto ifc : weights_iter) {
		weight = ifc->Weight();
		weights_store.write(reinterpret_cast<char *>(&weight), sizeof(float));
	}
	weights_store.close();
}