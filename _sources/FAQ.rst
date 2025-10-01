Frequently Asked Questions
==========================

**1. Why does RAM usage increase when creating multiple BayesMBAR instances?**

When creating multiple BayesMBAR instances in a loop, you may notice that RAM usage increases continuously. This is caused by JAX caching jitted functions.

**Solution:** To manage memory usage, periodically call ``jax.clear_caches()`` when creating many BayesMBAR instances:

.. code-block:: python

   import jax

   for i in range(many_iterations):
       # Create and use BayesMBAR instance
       mbar = BayesMBAR(...)
       # ... your code ...

       # Periodically clear JAX caches
       if i % 10 == 0:  # adjust frequency as needed
           jax.clear_caches()

**Note:** Clearing caches will affect all JAX jitted functions in your program, not just those from BayesMBAR.
