import { useState, useEffect } from "react";
import api from "../api"


function App() {
  const [pan, setPan] = useState(50);
  const [tilt, setTilt] = useState(50)

  useEffect(() => {
    const sendServoCommands = async () => {
      try {
        const response = api.post("/move/dual", {
          base: pan,
          tilt: tilt
        })
        console.log(response)
      } catch (error) {
        console.log(error)
      }

    }
    sendServoCommands()
  }, [pan, tilt])

  return (
    <>
      <section className="min-h-screen bg-gray-600 flex flex-col items-center justify-center">
        <div className="text-black text-5xl font-bold">Servo Slider</div>
        <div className="flex flex-col w-1/2 my-10">
          <div className="flex flex-col items-center">
            <h1 className="text-4xl my-2 font-bold">Base Servo</h1>
            <input
              type="range"
              min={0}
              max={100}
              value={pan}
              onChange={(e) => setPan(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-center text-lg font-bold ">{pan} %</span>
          </div>
          <div className="flex flex-col items-center">
            <h1 className="text-4xl my-2 font-bold">Tilt Servo</h1>
            <input
              type="range"
              min={0}
              max={100}
              value={tilt}
              onChange={(e) => setTilt(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-center text-lg font-bold ">{tilt} %</span>
          </div>
        </div>
      </section>
    </>
  );
}

export default App;
