module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      // Minimal configuration - no problematic plugins
    ],
  };
};
