package neurvolve

import (
	"encoding/json"
	"fmt"
	_ "github.com/couchbaselabs/logg"
	"github.com/gorilla/mux"
	ng "github.com/tleyden/neurgo"
	"net/http"
	"time"
)

func RegisterHandlers(pt *PopulationTrainer) {

	r := mux.NewRouter()

	showAllCortexes := func(w http.ResponseWriter, r *http.Request) {
		evaldPopulation := pt.GetPopulationSnapshot()
		marshalJson(evaldPopulation, w)
	}
	showAllCortexUuids := func(w http.ResponseWriter, r *http.Request) {
		evaldPopulation := pt.GetPopulationSnapshot()
		uuids := evaldPopulation.Uuids()
		marshalJson(uuids, w)
	}

	saveAllCortexes := func(w http.ResponseWriter, r *http.Request) {
		saveMap := make(map[string][]string)
		evaldPopulation := pt.GetPopulationSnapshot()
		for _, evaldCortex := range evaldPopulation {
			filename, filenameSvg, filenameFitness := saveCortex(evaldCortex)
			filenames := []string{filename, filenameSvg, filenameFitness}
			saveMap[evaldCortex.Cortex.NodeId.UUID] = filenames
		}
		marshalJson(saveMap, w)
	}

	showCortex := func(w http.ResponseWriter, r *http.Request) {
		evaldPopulation := pt.GetPopulationSnapshot()
		vars := mux.Vars(r)
		cortexUuid := vars["cortex_uuid"]
		evaldCortex := evaldPopulation.Find(cortexUuid)
		fmt.Fprintf(w, "%v", evaldCortex.Cortex)
	}

	saveCortex := func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		cortexUuid := vars["cortex_uuid"]
		evaldPopulation := pt.GetPopulationSnapshot()
		evaldCortex := evaldPopulation.Find(cortexUuid)
		filename, filenameSvg, filenameFitness := saveCortex(evaldCortex)
		fmt.Fprintf(w, "Json: %v Svg: %v Fit: %v", filename, filenameSvg, filenameFitness)
	}

	cortexSvgHandler := func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		cortexUuid := vars["cortex_uuid"]
		evaldPopulation := pt.GetPopulationSnapshot()
		evaldCortex := evaldPopulation.Find(cortexUuid)
		cortex := evaldCortex.Cortex
		cortex.RenderSVG(w)
	}

	r.HandleFunc("/", HomeHandler)
	r.HandleFunc("/cortex", showAllCortexes)
	r.HandleFunc("/cortex/uuid", showAllCortexUuids)
	r.HandleFunc("/cortex/save", saveAllCortexes)
	r.HandleFunc("/cortex/{cortex_uuid}", showCortex)
	r.HandleFunc("/cortex/{cortex_uuid}/save", saveCortex)
	r.HandleFunc("/cortex/{cortex_uuid}/svg", cortexSvgHandler)
	http.Handle("/", r)

}

func HomeHandler(w http.ResponseWriter, r *http.Request) {
	routeMap := make(map[string]string)
	routeMap["/cortex"] = "Show All Cortexes"
	routeMap["/cortex/uuid"] = "Show All Cortex Uuids"
	routeMap["/cortex/save"] = "Save All Cortexes to temp files"
	routeMap["/cortex/{cortex_uuid}"] = "Show Cortex for uuid"
	routeMap["/cortex/{cortex_uuid}/svg"] = "Show Cortex SVG for uuid"
	routeMap["/cortex/{cortex_uuid}/save"] = "Save single cortex to temp file"
	marshalJson(routeMap, w)
}

func marshalJson(v interface{}, w http.ResponseWriter) {
	json, err := json.Marshal(v)
	if err != nil {
		fmt.Fprintf(w, "Error marshaling json: %v", err)
	}
	_, err = w.Write(json)
	if err != nil {
		fmt.Fprintf(w, "Error writing response: %v", err)
	}
}

func saveCortex(evaldCortex EvaluatedCortex) (filename, filenameSvg, filenameFitness string) {
	unixTimestamp := time.Now().Unix()
	cortex := evaldCortex.Cortex
	uuid := cortex.NodeId.UUID
	filename = fmt.Sprintf("/tmp/%v-%v.json", uuid, unixTimestamp)
	cortex.MarshalJSONToFile(filename)
	filenameSvg = fmt.Sprintf("/tmp/%v-%v.svg", uuid, unixTimestamp)
	cortex.RenderSVGFile(filenameSvg)
	fitnessStr := fmt.Sprintf("%v", evaldCortex.Fitness)
	filenameFitness = fmt.Sprintf("/tmp/%v-%v-%v.fit", uuid, unixTimestamp, fitnessStr)
	ng.WriteStringToFile(fitnessStr, filenameFitness)
	return
}
