import React, { useState, useEffect } from 'react';

interface Product {
  id: string;
  name: string;
  price: number;
  stock: number;
}

interface CartItem extends Product {
  quantity: number;
}

const POSTerminal: React.FC = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [cart, setCart] = useState<CartItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  // Fetch products from your backend
  useEffect(() => {
    const fetchProducts = async () => {
      try {
        const response = await fetch('/api/products'); // Replace with your actual API endpoint
        const data = await response.json();
        setProducts(data);
      } catch (error) {
        console.error('Error fetching products:', error);
      }
    };

    fetchProducts();
  }, []);

  const addToCart = (product: Product) => {
    if (product.stock <= 0) {
      alert('Product out of stock!');
      return;
    }

    const existingItem = cart.find(item => item.id === product.id);
    if (existingItem) {
      if (existingItem.quantity >= product.stock) {
        alert('Not enough stock!');
        return;
      }
      setCart(cart.map(item =>
        item.id === product.id
          ? { ...item, quantity: item.quantity + 1 }
          : item
      ));
    } else {
      setCart([...cart, { ...product, quantity: 1 }]);
    }
  };

  const removeFromCart = (productId: string) => {
    setCart(cart.filter(item => item.id !== productId));
  };

  const updateQuantity = (productId: string, newQuantity: number) => {
    const product = products.find(p => p.id === productId);
    if (!product || newQuantity > product.stock) {
      alert('Invalid quantity!');
      return;
    }

    setCart(cart.map(item =>
      item.id === productId
        ? { ...item, quantity: newQuantity }
        : item
    ));
  };

  const calculateTotal = () => {
    return cart.reduce((total, item) => total + (item.price * item.quantity), 0);
  };

  const handleCheckout = async () => {
    try {
      // TODO: Implement API call to process sale and update stock
      // Example:
      // await processSale({
      //   items: cart,
      //   total: calculateTotal(),
      //   timestamp: new Date()
      // });
      
      setCart([]);
      alert('Sale completed successfully!');
    } catch (error) {
      alert('Error processing sale!');
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      {/* Products Section */}
      <div className="col-span-8">
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
              product.name.toLowerCase().includes(searchTerm.toLowerCase())
            )
            .map(product => (
              <div
                key={product.id}
                className="p-4 border rounded cursor-pointer hover:bg-gray-50"
                onClick={() => addToCart(product)}
              >
                <h3 className="font-bold">{product.name}</h3>
                <p>${product.price.toFixed(2)}</p>
                <p>Stock: {product.stock}</p>
              </div>
            ))}
        </div>
      </div>

      {/* Cart Section */}
      <div className="col-span-4 border-l p-4">
        <h2 className="text-xl font-bold mb-4">Cart</h2>
        {cart.map(item => (
          <div key={item.id} className="mb-2 p-2 border rounded">
            <div className="flex justify-between">
              <span>{item.name}</span>
              <button
                onClick={() => removeFromCart(item.id)}
                className="text-red-500"
              >
                ×
              </button>
            </div>
            <div className="flex justify-between items-center mt-2">
              <input
                type="number"
                min="1"
                max={item.stock}
                value={item.quantity}
                onChange={(e) => updateQuantity(item.id, parseInt(e.target.value))}
                className="w-20 p-1 border rounded"
              />
              <span>${(item.price * item.quantity).toFixed(2)}</span>
            </div>
          </div>
        ))}
        <div className="mt-4 border-t pt-4">
          <div className="flex justify-between text-xl font-bold">
            <span>Total:</span>
            <span>${calculateTotal().toFixed(2)}</span>
          </div>
          <button
            onClick={handleCheckout}
            disabled={cart.length === 0}
            className="w-full mt-4 p-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
          >
            Checkout
          </button>
        </div>
      </div>
    </div>
  );
};

export default POSTerminal;