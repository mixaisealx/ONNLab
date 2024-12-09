#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"

#include <cstdlib>
#include <concepts>
#include <vector>
#include <span>
#include <functional>

namespace nn
{
	template<typename T>
	concept ConnectionInherit = std::is_base_of<nn::interfaces::CBI, T>::value;

	template<ConnectionInherit ConnectionT>
	class DenseLayerStaticConnectomHolder {
		ConnectionT *storage_base, *storage_end;
		std::span<ConnectionT> obj_view;
	public:
		using ConnectionEmplacer = void(ConnectionT*const mem_ptr, nn::interfaces::NBI *from, nn::interfaces::NBI *to);

		DenseLayerStaticConnectomHolder(nn::interfaces::BasicLayerInterface *layer1,
								  nn::interfaces::BasicLayerInterface *layer2,
								  std::function<ConnectionEmplacer> emplacer) {
			
			size_t connections_count = layer1->Neurons().size() * layer2->Neurons().size();
			storage_base = reinterpret_cast<ConnectionT*>(malloc(connections_count * sizeof(ConnectionT))); // Yes, I know about possible nullptr, BUT it is only lab code
			storage_end = storage_base + connections_count;

			ConnectionT *mem_ptr = storage_base;
			for (auto from : layer1->Neurons()) {
				for (auto to : layer2->Neurons()) {
					emplacer(mem_ptr, from, to);
					++mem_ptr;
				}
			}
			obj_view = std::span<ConnectionT>{ storage_base, connections_count };
		}

		const std::span<ConnectionT> &ConnectionsInside() {
			return obj_view;
		}

		~DenseLayerStaticConnectomHolder() {
			for (ConnectionT *curr = storage_base; curr != storage_end; ++curr) {
				dynamic_cast<nn::interfaces::CBI *>(curr)->~ConnectionBasicInterface();
			}
			free(storage_base);
		}
	};
}