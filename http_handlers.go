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

func RegisterHandlers(population CortexPopulation) {

	r := mux.NewRouter()

	showAllCortexes := func(w http.ResponseWriter, r *http.Request) {
		marshalJson(population, w)
	}

	showAllCortexUuids := func(w http.ResponseWriter, r *http.Request) {
		uuids := population.Uuids()
		marshalJson(uuids, w)
	}

	saveAllCortexes := func(w http.ResponseWriter, r *http.Request) {
		saveMap := make(map[string][]string)
		for _, cortex := range population {
			filename, filenameSvg := saveCortex(cortex)
			filenames := []string{filename, filenameSvg}
			saveMap[cortex.NodeId.UUID] = filenames
		}
		marshalJson(saveMap, w)
	}

	showCortex := func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		cortexUuid := vars["cortex_uuid"]
		cortex := population.Find(cortexUuid)
		fmt.Fprintf(w, "%v", cortex)
	}

	saveCortex := func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		cortexUuid := vars["cortex_uuid"]
		cortex := population.Find(cortexUuid)
		filename, filenameSvg := saveCortex(cortex)
		fmt.Fprintf(w, "Json: %v Svg: %v", filename, filenameSvg)
	}

	cortexSvgHandler := func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		cortexUuid := vars["cortex_uuid"]
		cortex := population.Find(cortexUuid)
		cortex.RenderSVG(w)
	}

	r.HandleFunc("/", HomeHandler)
	r.HandleFunc("/cortex", showAllCortexes)
	r.HandleFunc("/cortex/uuid", showAllCortexUuids)
	r.HandleFunc("/cortex/tmpfile", saveAllCortexes)
	r.HandleFunc("/cortex/{cortex_uuid}", showCortex)
	r.HandleFunc("/cortex/{cortex_uuid}/svg", cortexSvgHandler)
	r.HandleFunc("/cortex/{cortex_uuid}/tmpfile", saveCortex)
	http.Handle("/", r)

}

func HomeHandler(w http.ResponseWriter, r *http.Request) {
	routeMap := make(map[string]string)
	routeMap["/cortex"] = "Show All Cortexes"
	routeMap["/cortex/uuid"] = "Show All Cortex Uuids"
	routeMap["/cortex/tmpfile"] = "Save All Cortexes to temp files"
	routeMap["/cortex/{cortex_uuid}"] = "Show Cortex for uuid"
	routeMap["/cortex/{cortex_uuid}/svg"] = "Show Cortex SVG for uuid"
	routeMap["/cortex/{cortex_uuid}/tmpfile"] = "Save single cortex to temp file"
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

func saveCortex(cortex *ng.Cortex) (filename string, filenameSvg string) {
	unixTimestamp := time.Now().Unix()
	uuid := cortex.NodeId.UUID
	filename = fmt.Sprintf("/tmp/%v-%v.json", uuid, unixTimestamp)
	cortex.MarshalJSONToFile(filename)
	filenameSvg = fmt.Sprintf("/tmp/%v-%v.svg", uuid, unixTimestamp)
	cortex.RenderSVGFile(filenameSvg)
	return
}
