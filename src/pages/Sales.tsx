import { useEffect, useState } from 'react';
import PageBreadcrumb from "../components/common/PageBreadCrumb";
import PageMeta from "../components/common/PageMeta";
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import Badge from "../components/ui/badge/Badge";

interface Sale {
  id: string;
  userId: string;
  orderDate: string;
  totalAmount: number;
}

export default function Sales() {
  const [sales, setSales] = useState<Sale[]>([]);

  useEffect(() => {
    const fetchSales = async () => {
      try {
        const response = await fetch('data/Sales.csv');
        const csvText = await response.text();
        
        // Skip header row and parse CSV
        const rows = csvText.split('\n').slice(1);
        const parsedSales = rows
          .map(row => {
            const [id, userId, orderDate, totalAmount] = row.split(',');
            return {
              id,
              userId,
              orderDate,
              totalAmount: parseFloat(totalAmount)
            };
          })
          // Sort by order date in descending order (latest first)
          .sort((a, b) => new Date(b.orderDate).getTime() - new Date(a.orderDate).getTime());
        
        setSales(parsedSales);
      } catch (error) {
        console.error('Error loading sales data:', error);
        setSales([]); // Set empty array on error
      }
    };

    fetchSales();
  }, []);
  return (
    <>
      <PageMeta
        title="Sales History | POPWARUNG Dashboard"
        description="Sales history page for POPWARUNG Dashboard"
      />
      <PageBreadcrumb pageTitle="Sales History" />

      <div className="rounded-2xl border border-gray-200 bg-white px-4 pb-3 pt-4 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6">
        <div className="flex flex-col gap-2 mb-4 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
            Sales History
          </h3>

          <div className="flex items-center gap-3">
            <button className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-theme-sm font-medium text-gray-700 shadow-theme-xs hover:bg-gray-50 hover:text-gray-800 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-white/[0.03] dark:hover:text-gray-200">
              Filter
            </button>
            <button className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2.5 text-theme-sm font-medium text-white shadow-theme-xs hover:bg-red-700">
              Export
            </button>
          </div>
        </div>

        <div className="max-w-full overflow-x-auto">
          <Table>
            <TableHeader className="border-gray-100 dark:border-gray-800 border-y">
              <TableRow>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Sale ID
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  User ID
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Order Date
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Total Amount (MYR)
                </TableCell>
              </TableRow>
            </TableHeader>

            <TableBody className="divide-y divide-gray-100 dark:divide-gray-800">
              {sales.map((sale) => (
                <TableRow key={sale.id}>
                  <TableCell className="py-3 text-gray-800 dark:text-white/90">
                    {sale.id}
                  </TableCell>
                  <TableCell className="py-3 text-gray-800 dark:text-white/90">
                    {sale.userId}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    {sale.orderDate}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    RM {sale.totalAmount.toFixed(2)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </>
  );
}