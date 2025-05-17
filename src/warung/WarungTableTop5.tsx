"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { Dropdown } from "../components/ui/dropdown/Dropdown";
import { DropdownItem } from "../components/ui/dropdown/DropdownItem";
import { MoreDotIcon } from "../icons";
import { useEffect, useState, useRef } from "react";
import rawSalesData from "../../dataset_script/sales_output_new.json";

interface RawSaleEntry {
  Date: string;
  ProductID: string;
  StoreID: string;
  UserID: string;
  UnitsSold: number;
  "Price(OfProduct)": number;
  TotalEarn: number;
}

interface AggregatedProduct {
  id: string;
  totalEarning: number;
}

type TimeFilter = "today" | "month" | "year";

export default function Top5SellingProducts() {
  const sales = rawSalesData as RawSaleEntry[];
  const uniqueStores = Array.from(new Set(sales.map((s) => s.StoreID))).sort();

  const [selectedStores, setSelectedStores] = useState<string[]>(uniqueStores);
  const [timeFilter, setTimeFilter] = useState<TimeFilter>("today");
  const [topProducts, setTopProducts] = useState<AggregatedProduct[]>([]);

  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const toggleDropdown = () => setIsDropdownOpen((open) => !open);
  const closeDropdown = () => setIsDropdownOpen(false);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        closeDropdown();
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleTimeFilterSelect = (filter: TimeFilter) => {
    setTimeFilter(filter);
  };

  const handleCheckboxChange = (storeId: string) => {
    setSelectedStores((prev) =>
      prev.includes(storeId) ? prev.filter((s) => s !== storeId) : [...prev, storeId]
    );
  };

  useEffect(() => {
    const now = new Date();
    const filteredByTime = sales.filter((entry) => {
      const date = new Date(entry.Date);
      if (timeFilter === "today") {
        return (
          date.getFullYear() === now.getFullYear() &&
          date.getMonth() === now.getMonth() &&
          date.getDate() === now.getDate()
        );
      } else if (timeFilter === "month") {
        return date.getFullYear() === now.getFullYear() && date.getMonth() === now.getMonth();
      } else if (timeFilter === "year") {
        return date.getFullYear() === now.getFullYear();
      }
      return true;
    });

    const filtered = filteredByTime.filter((entry) => selectedStores.includes(entry.StoreID));
    const earningsMap = new Map<string, number>();
    filtered.forEach((entry) => {
      const currentTotal = earningsMap.get(entry.ProductID) || 0;
      earningsMap.set(entry.ProductID, currentTotal + entry.TotalEarn);
    });

    const aggregated: AggregatedProduct[] = Array.from(earningsMap.entries()).map(
      ([id, totalEarning]) => ({ id, totalEarning })
    );

    const sorted = aggregated.sort((a, b) => b.totalEarning - a.totalEarning).slice(0, 5);
    setTopProducts(sorted);
  }, [selectedStores, timeFilter, sales]);

  const maxEarning = topProducts.reduce((max, p) => (p.totalEarning > max ? p.totalEarning : max), 0);

  const timeFilterLabels: Record<TimeFilter, string> = {
    today: "Today",
    month: "This Month",
    year: "This Year",
  };

  return (
    <div className="space-y-4">
      {/* Table */}
      <div className="relative overflow-visible rounded-xl border border-gray-200 bg-white dark:border-white/[0.05] dark:bg-white/[0.03]">
        {/* 3-dot Dropdown Menu (top-right corner) */}
        <div className="absolute top-3 right-3 z-20 overflow-visible" ref={dropdownRef}>
          <button className="dropdown-toggle" onClick={toggleDropdown}>
            <MoreDotIcon className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 size-6" />
          </button>
          <Dropdown
            isOpen={isDropdownOpen}
            onClose={closeDropdown}
            className="w-52 p-2"
          >
            <div className="border-b pb-2 mb-2 text-sm text-gray-500 dark:text-gray-400">
              Time Filter
            </div>
            {(["today", "month", "year"] as TimeFilter[]).map((filter) => (
              <DropdownItem
                key={filter}
                onItemClick={() => handleTimeFilterSelect(filter)}
                className={`flex w-full font-normal text-left rounded-lg ${
                  timeFilter === filter
                    ? "text-red-600 font-semibold"
                    : "text-gray-500"
                } hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300`}
              >
                {timeFilterLabels[filter]}
              </DropdownItem>
            ))}

            <div className="border-t pt-2 mt-2 text-sm text-gray-500 dark:text-gray-400">
              Stores
            </div>
            {uniqueStores.map((storeId) => (
              <DropdownItem
                key={storeId}
                onItemClick={() => handleCheckboxChange(storeId)}
                className="flex items-center gap-2 font-normal text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
              >
                <input
                  type="checkbox"
                  checked={selectedStores.includes(storeId)}
                  onChange={() => handleCheckboxChange(storeId)}
                  className="accent-red-600"
                  onClick={(e) => e.stopPropagation()}
                />
                {storeId}
              </DropdownItem>
            ))}
          </Dropdown>
        </div>

        {/* Table Content */}
        <div className="max-w-full overflow-x-auto min-h-[38px]">
          <Table>
            <TableHeader className="border-b border-gray-100 dark:border-white/[0.05]">
              <TableRow>
                <TableCell
                  isHeader
                  className="px-5 py-3 text-start text-theme-xs font-medium text-gray-500 dark:text-gray-400"
                >
                  Product ID
                </TableCell>
                <TableCell
                  isHeader
                  className="px-5 py-3 text-start text-theme-xs font-medium text-gray-500 dark:text-gray-400"
                >
                  Total Earning (RM)
                </TableCell>
              </TableRow>
            </TableHeader>
            <TableBody className="divide-y divide-gray-100 dark:divide-white/[0.05]">
              {topProducts.map((product) => {
                const barWidthPercent = maxEarning > 0 ? (product.totalEarning / maxEarning) * 100 : 0;

                return (
                  <TableRow key={product.id}>
                    <TableCell className="px-5 py-4 text-start">{product.id}</TableCell>
                    <TableCell className="px-5 py-4 text-start w-[160px]">
                      <div className="relative h-6 bg-red-200 dark:bg-red-900 rounded overflow-hidden">
                        <div
                          className="absolute left-0 top-0 h-full bg-red-600 dark:bg-red-400 transition-all"
                          style={{ width: `${barWidthPercent}%` }}
                        ></div>
                        <div className="relative z-10 px-2 text-white font-semibold whitespace-nowrap">
                          RM {product.totalEarning.toFixed(2)}
                        </div>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
}
