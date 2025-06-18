#pragma once

#include <vector>
#include <memory>

template<typename IteratorValueType>
class IterableAggregation {
	struct ItemBase {
		virtual ~ItemBase() = default;
		virtual bool IsBusy() const = 0;
		virtual IteratorValueType CurrentValue() = 0;
		virtual bool Next() = 0;
	};

	template<typename IteratorT, typename IterableT>
	struct Item : public ItemBase {
		IterableT &object;
		Item(IterableT &object):object(object) {}

		~Item() override {
			iterators.reset();
		}

		std::unique_ptr<std::pair<IteratorT, IteratorT>> iterators;
		bool IsBusy() const override {
			return static_cast<bool>(iterators);
		}

		bool Next() override {
			if (iterators) {
				++(iterators->first);
				if (iterators->first == iterators->second) {
					iterators.reset();
					return false;
				}
			} else {
				iterators = std::unique_ptr<std::pair<IteratorT, IteratorT>>(new std::pair<IteratorT, IteratorT>{ std::begin(object), std::end(object) });
				if (iterators->first == iterators->second) {
					iterators.reset();
					return false;
				}
			}
			return true;
		}
		IteratorValueType CurrentValue() override {
			return (IteratorValueType)(*(iterators->first));
		}
	};

	std::vector<ItemBase *> items;

	unsigned appends_count;

	IterableAggregation(const IterableAggregation &) = delete;
	IterableAggregation &operator=(const IterableAggregation &) = delete;
public:
	IterableAggregation() {
		appends_count = 0;
	}

	template<typename IteratorT, typename IterableT>
	void AppendIterableItem(IterableT &object) {
		++appends_count;
		items.push_back(new Item<IteratorT, IterableT>(object));
	}

	void Reset() {
		appends_count = 0;
		for (ItemBase *itm : items) {
			delete itm;
		}
		items.clear();
	}

	~IterableAggregation() {
		for (ItemBase* itm : items) {
			delete itm;
		}
	}

	struct Iterator {
		using iterator_category = std::forward_iterator_tag;
		using value_type = IteratorValueType;
		using pointer = IteratorValueType *;
		using reference = IteratorValueType &;

		IteratorValueType operator*() {
			if (end) throw std::exception("End have no value!");
			if (parent_appends_count != parent->appends_count) throw std::exception("Iterator is invalid!");
			return item->CurrentValue();
		}
		Iterator &operator++() {
			if (end) throw std::exception("Already on the end!");
			if (parent_appends_count != parent->appends_count) throw std::exception("Iterator is invalid!");
			
			if (item->Next()) {
				return *this;
			} else if (item_idx == parent_appends_count) {
				this->end = true;
				
			}
			while (true) {
				if (++item_idx == parent_appends_count) {
					this->end = true;
					return *this;
				}
				item = parent->items[item_idx];
				if (item->IsBusy()) {
					throw std::exception("Some iterators are already in use!");
				}
				if (item->Next()) {
					return *this;
				} 
			}
		}
		Iterator operator++(int) {
			auto tmp = *this;
			++*this;
			return tmp;
		}
		bool operator==(const Iterator &other) const {
			if (parent == other.parent && end == other.end) {
				return true;
			}
			return false;
		}
		bool operator!=(const Iterator &other) const {
			return !(*this == other);
		}
	private:
		friend class IterableAggregation;

		Iterator(IterableAggregation *parent, bool end): parent(parent), end(end) {
			parent_appends_count = parent->appends_count;
			item_idx = std::numeric_limits<unsigned>::max(); // To prepare flip to 0
			item = nullptr;
			if (!end) {
				while (true) {
					if (++item_idx == parent_appends_count) {
						this->end = true;
						break;
					}
					item = parent->items[item_idx];
					if (item->IsBusy()) {
						throw std::exception("Some iterators are already in use!");
					}
					if (item->Next()) {
						break;
					}
				}
			} 
		}

		IterableAggregation *parent;
		ItemBase *item;
		unsigned parent_appends_count;
		unsigned item_idx;
		bool end;
	};

	Iterator begin() {
		return Iterator(this, false);
	}

	Iterator end() {
		return Iterator(this, true);
	}
};