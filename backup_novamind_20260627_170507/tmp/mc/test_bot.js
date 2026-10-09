const mineflayer = require('mineflayer')
const bot = mineflayer.createBot({ host:'localhost', port:25565, username:'Nova_1', version:'1.20.4', auth:'offline' })
bot.on('spawn', () => {
  console.log('SPAWNED pos=', bot.entity.position.toString(), 'hp=', bot.health, 'food=', bot.food)
  bot.chat('Nova_1 acordou')
  let n=0
  const iv=setInterval(()=>{ bot.setControlState('forward',true); setTimeout(()=>bot.setControlState('forward',false),400); n++; if(n>=3){clearInterval(iv); console.log('ANDOU ok pos=',bot.entity.position.toString()); bot.quit(); process.exit(0)} },1500)
})
bot.on('error', e => { console.log('ERR', e.message); process.exit(1) })
bot.on('kicked', r => { console.log('KICKED', JSON.stringify(r)); process.exit(1) })
setTimeout(()=>{ console.log('TIMEOUT sem spawn'); process.exit(2) }, 30000)
