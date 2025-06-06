#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicWghOptI.h"
#include "OptimizerI.h"
#include "BatchNeuronBasicI.h"

namespace nn
{
	template<interfaces::OptimizerInherit OptimizerT>
	class HyperConnection {
		HyperConnection(const HyperConnection &) = delete;
		HyperConnection &operator=(const HyperConnection &) = delete;
	public:
		class Connection : public interfaces::ConnectionBasicInterface, public interfaces::BasicWeightOptimizableInterface {
				interfaces::NeuronBasicInterface *from;
				interfaces::NeuronBasicInterface *to;
				HyperConnection *superobj;
				float gradient;

				Connection(const Connection &) = delete;
				Connection &operator=(const Connection &) = delete;
			public:
				friend class HyperConnection;

				Connection(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to, HyperConnection *superobj): from(from), to(to), superobj(superobj) {
					interfaces::BatchNeuronBasicI *bfrom = dynamic_cast<interfaces::BatchNeuronBasicI *>(from);
					interfaces::BatchNeuronBasicI *bto = dynamic_cast<interfaces::BatchNeuronBasicI *>(to);
					unsigned fromb = (bfrom ? bfrom->GetCurrentBatchSize() : 1);
					unsigned tob = (bto ? bto->GetCurrentBatchSize() : 1);
					if (fromb != std::numeric_limits<unsigned>::max() &&
						tob != std::numeric_limits<unsigned>::max() &&
						fromb != tob) {
						throw std::exception("Different batch sizes is not allowed!");
					}
					NBI_AddOutputConnection(from, this);
					NBI_AddInputConnection(to, this);
					superobj->connections.push_back(this);
					gradient = 0;
				}

				~Connection() override {
					if (from && to) {
						NBI_RemoveOutputConnection(from, this);
						NBI_RemoveInputConnection(to, this);
						from = to = nullptr;
					}
				}

				interfaces::NeuronBasicInterface *From() override {
					return from;
				}

				interfaces::NeuronBasicInterface *To() override {
					return to;
				}

				float Weight() override {
					return superobj->Weight();
				}

				void Weight(float weight) override {
					throw std::exception("Changing hyper-connection weight from the part is not safe so is not allowed!");
				}

				void WeightOptimReset() override {
					throw std::exception("Resetting hyper-connection optimizer from the part is not safe so is not allowed!");
				}

				void WeightOptimDoUpdate(float gradient) override {
					this->gradient = gradient;
					if (++superobj->connections_optim_done == superobj->connections.size()) {
						superobj->WeightOptimDoUpdate();
					}
				}
		};

		HyperConnection(unsigned connections_count, OptimizerT *optimizer, float initial_weight = 0.0f): optimizer(optimizer), weight(initial_weight) {
			connections_optim_done = 0;
			connections.reserve(connections_count);
		}

		float Weight() const {
			return weight;
		}

		void Weight(float weight) {
			this->weight = weight;
		}

		void WeightOptimReset() {
			optimizer->Reset(&optimizer_context);
		}

		void WeightOptimDoUpdate() {
			connections_optim_done = 0;
			float gradient = 0;
			for (auto conn : connections) {
				gradient += conn->gradient;
			}
			gradient /= connections.size();
			weight -= optimizer->CalcDelta(gradient, &optimizer_context);
		}

	private:
		OptimizerT *optimizer;
		float weight;
		unsigned connections_optim_done;
		OptimizerT::State optimizer_context;
		std::vector<Connection *> connections;
		
	};
}
