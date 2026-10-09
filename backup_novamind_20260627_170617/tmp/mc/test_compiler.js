const wm = require('./worldmodel');
const lang = require('./language');

console.log("=== Testing Cultural Compiler JS ===");

// Inject a rule
lang.injectCulturalRule("agua contaminada", "doenca");

// Test comprehend
const bot = {brain: {}};
const result = lang.comprehend(bot, "Encontrei agua contaminada ali");

console.log("Comprehend result for 'agua contaminada':", result);

wm.injectRule("agua contaminada", "beber", -5);
console.log("Predict beber agua contaminada:", wm.predict("agua contaminada", "beber"));

console.log("=== DONE ===");
