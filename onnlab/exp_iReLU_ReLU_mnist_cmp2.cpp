#include "onnlab.h"

#include "NNB_Connection.h"
#include "OptimizerGD.h"
#include "OptimizerAdam.h"
#include "NNB_Linear.h"
#include "NNB_ReLU.h"
#include "NNB_iReLU.h"
#include "NNB_Input.h"
#include "NNB_ConstInput.h"
#include "NNB_Layer.h"
#include "NNB_LinearSlim.h"
#include "LearnGuiderFwBPgThreadAble.h"
#include "DenseLayerStaticConnectomHolder.h"
#include "NeuronHoldingStaticLayer.h"

#include "CSVreader.h"
#include "NetQualityCalcUtils.h"

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
#include <typeinfo>

static unsigned THREADS_COUNT = 16;
static const unsigned LOG_INTERVAL = 400;

static const unsigned BATCH_SIZE = 64;
static const unsigned EPOCHS = 10;
static const float VALID_TEST_SPLIT_RATIO = 0.1f; // sizeof(validation)/sizeof(full testing dataset size), should be small to run fast

static const float LEARNING_RATE = 0.001f;

static const unsigned LAYER_HIDDEN1_SIZE = 32;
static const unsigned LAYER_HIDDEN2_SIZE = 16;

struct MNIST_entry {
	MNIST_entry(uint8_t label, std::vector<uint8_t> input, uint16_t reseve):label(label), input(input) {
		input.reserve(reseve);
	}
	uint8_t label;
	std::vector<uint8_t> input;
};

template<typename nnReLU>
static inline std::tuple<float, float> MNIST_Train_setup(const std::vector<MNIST_entry> &mnist_train, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test);

static unsigned random_seed;

void exp_iReLU_ReLU_mnist_cmp2() {
	std::cout << "exp_iReLU_ReLU_mnist_cmp2" << std::endl;

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

	unsigned loop_id = 0;
	while (true) {
		random_seed = randevice();
		std::cout << "===================================================\n";
		std::cout << "================ NEW TRAINING LOOP ================\n";
		std::cout << "===================================================\n";
		std::cout << "LOOP INDEX: " << ++loop_id << '\n';
		std::cout << "RANDOM SEED: " << random_seed << std::endl;
		std::cout << "===================================================\n";

		std::mt19937 preudorandom(random_seed);

		auto Trainer_ReLU = [&]() {
			auto [relu_acc, relu_f1] = MNIST_Train_setup<nn::NNB_ReLUb<BATCH_SIZE, true>>(mnist_train, mnist_valid, mnist_test, mnist_train_classes_w.GetWeights(), mnist_test_classes_w.GetWeights());
			std::ofstream log("exp_iReLU_ReLU_mnist_cmp2_relu.log", std::ios::app);
			log << relu_acc << '\t' << relu_f1 << '\n';
			log.close();
		};

		auto Trainer_iReLU = [&]() {
			auto [irelu_acc, irelu_f1] = MNIST_Train_setup<nn::NNB_iReLUb<BATCH_SIZE, true>>(mnist_train, mnist_valid, mnist_test, mnist_train_classes_w.GetWeights(), mnist_test_classes_w.GetWeights());
			std::ofstream log("exp_iReLU_ReLU_mnist_cmp2_irelu.log", std::ios::app);
			log << irelu_acc << '\t' << irelu_f1 << '\n';
			log.close();
		};

		std::thread wrk_relu(Trainer_ReLU);
		std::thread wrk_irelu(Trainer_iReLU);

		wrk_relu.join();
		wrk_irelu.join();
	}
}

template <typename T>
struct TypeName {
	static const char *Get() {
		return typeid(T).name();
	}
};

template<typename nnReLU>
static inline std::tuple<float, float> MNIST_Train_setup(const std::vector<MNIST_entry> &mnist_train_sample, const std::vector<MNIST_entry> &mnist_valid, const std::vector<MNIST_entry> &mnist_test, const std::vector<float> &weights_train, const std::vector<float> &weights_test) {
	std::mt19937 preudorandom(random_seed);
	std::uniform_real_distribution<float> randistributor(0.01f, 0.1f);

	const unsigned INPUT_SIZE = 28 * 28;

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

	std::osyncstream(std::cout) << TypeName<nnReLU>::Get() << ": Training... " << '\n';

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
				std::osyncstream(std::cout) << TypeName<nnReLU>::Get() << std::fixed << std::setprecision(3) << ": Epoch: " << (epoch + 1) << "  Batch: " << std::setfill('0') << std::setw(3) << batch_iterations << '/' << batch_iterations_all << "  Loss: " << loss << "  Accuracy(local): " << accuracy << "%  F1(local): " << f1 << "\n";
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
		std::osyncstream(std::cout) << TypeName<nnReLU>::Get() << std::fixed << std::setprecision(3) << ": Accuracy (validation dataset): " << f1score_test_calculator.CalcAccuracy() << "%  F1: " << f1score_test_calculator.CalcF1() << "\n";
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


	std::osyncstream(std::cout) << TypeName<nnReLU>::Get() << ": Accuracy calculating...\n";

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
			std::osyncstream(std::cout) << TypeName<nnReLU>::Get() << ": Batch: " << batch_count << '/' << batch_iterations_all << '\n';
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
	std::osyncstream(std::cout) << TypeName<nnReLU>::Get() << std::fixed << std::setprecision(3) << ": Accuracy (test dataset): " << accuracy << "%  F1: " << f1 << "\n";

	f1score_test_calculator.Reset();

	for (auto &thr : workers) {
		thr.join();
	}
	return std::tuple(accuracy, f1);
}