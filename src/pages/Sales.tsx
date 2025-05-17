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

interface SaleItem {
  productId: string;
  productName: string;
  quantity: number;
  price: number;
}

interface Sale {
  id: string;
  timestamp: string;
  items: SaleItem[];
  total: number;
  status: "Completed" | "Refunded" | "Cancelled";
}

// Sample data - replace with actual data later
const sales: Sale[] = [
  {
    id: "S001",
    timestamp: "2024-01-20 14:30:00",
    items: [
      {
        productId: "P001",
        productName: "Product 1",
        quantity: 2,
        price: 29.99
      }
    ],
    total: 59.98,
    status: "Completed"
  }
];

export default function Sales() {
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
                  Date & Time
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Items
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Total
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Status
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
                    {sale.timestamp}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    {sale.items.length} items
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    RM {sale.total.toFixed(2)}
                  </TableCell>
                  <TableCell className="py-3">
                    <Badge
                      size="sm"
                      color={
                        sale.status === "Completed"
                          ? "success"
                          : sale.status === "Refunded"
                          ? "warning"
                          : "error"
                      }
                    >
                      {sale.status}
                    </Badge>
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