import React, { useState, useEffect } from 'react';
import Button from '../ui/button/Button'; // Assuming Button component path
import { Modal } from '../ui/modal'; // Correct Modal component import path

interface Product {
  productID: string;
  product_name: string;
  description: string;
  CurrentStock: number;
  ReorderPoint: number;
  SafetyStock: number;
  LeadTime: number;
  cost: number;
}

const POSTerminal: React.FC = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [quantityToSell, setQuantityToSell] = useState<number>(1);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        const response = await fetch('http://localhost:3334/product');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setProducts(data);
      } catch (error) {
        console.error('Error fetching products:', error);
      }
    };

    fetchProducts();
  }, []);

  const handleAddToCartClick = (product: Product) => {
    setSelectedProduct(product);
    setIsModalOpen(true);
    setQuantityToSell(1); // Reset quantity when opening modal
    setError(null); // Clear previous errors
  };

  const handleQuantityChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const quantity = parseInt(event.target.value, 10);
    if (!isNaN(quantity) && quantity > 0) {
      setQuantityToSell(quantity);
      if (selectedProduct && quantity > selectedProduct.CurrentStock) {
        setError("Quantity cannot exceed current stock.");
      } else {
        setError(null);
      }
    } else {
      setQuantityToSell(1);
      setError("Please enter a valid quantity.");
    }
  };

  const handleConfirmSale = () => {
    if (selectedProduct && quantityToSell > 0 && quantityToSell <= selectedProduct.CurrentStock) {
      // Implement sale logic here (user will handle API update)
      console.log(`Selling ${quantityToSell} of ${selectedProduct.product_name}`);
      // For now, just close the modal
      setIsModalOpen(false);
    } else if (selectedProduct) {
        setError("Invalid quantity or insufficient stock.");
    }
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedProduct(null);
    setQuantityToSell(1);
    setError(null);
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      {/* Products Section */}
      <div className="col-span-12"> {/* Use full width as there's no cart section */}
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search products..."
            className="w-full p-2 border rounded"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="grid grid-cols-3 gap-4">
          {products
            .filter(product =>
              product.product_name.toLowerCase().includes(searchTerm.toLowerCase())
            )
            .map(product => (
              <div
                key={product.productID}
                className="p-4 border rounded cursor-pointer hover:bg-gray-50"
              >
                <h3 className="font-bold">{product.product_name}</h3>
                <p>ID: {product.productID}</p>
                <p>Stock: {product.CurrentStock}</p>
                <p>Reorder Point: {product.ReorderPoint}</p>
                <p>Safety Stock: {product.SafetyStock}</p>
                <p>${product.cost.toFixed(2)}</p>
                <Button
                  className="mt-2 w-full" // Apply full width class
                  onClick={() => handleAddToCartClick(product)}
                  disabled={product.CurrentStock <= 0}
                  variant="primary" // Use variant for primary style
                >
                  Add to Cart
                </Button>
              </div>
            ))}
        </div>
      </div>

      {/* Sale Modal */}
      <Modal isOpen={isModalOpen} onClose={handleCloseModal}>
        {selectedProduct && (
          <div className="fixed inset-0 flex items-center justify-center bg-black/50 z-50">
            <div className="bg-white rounded-lg p-6 w-96 text-center dark:bg-gray-800">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90 mb-4">
                Sell {selectedProduct.product_name}
              </h3>
              <div className="mb-4">
                <label htmlFor="quantity" className="block text-theme-sm font-medium text-gray-700 dark:text-gray-400">
                  Quantity to Sell:
                </label>
                <input
                  type="number"
                  id="quantity"
                  value={quantityToSell}
                  onChange={handleQuantityChange}
                  min="1"
                  max={selectedProduct.CurrentStock}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white/90"
                />
                {error && <p className="text-red-500 text-theme-sm mt-1">{error}</p>}
              </div>
              <div className="flex justify-center gap-3">
                <Button onClick={handleCloseModal} variant="outline">
                  Cancel
                </Button>
                <Button
                  onClick={handleConfirmSale}
                  disabled={!!error || quantityToSell <= 0 || quantityToSell > selectedProduct.CurrentStock}
                  className="bg-green-600 text-white shadow-theme-xs hover:bg-green-700 disabled:opacity-50"
                >
                  Confirm Sale
                </Button>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default POSTerminal;
